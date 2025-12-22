from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import os
import re
import difflib
import unicodedata

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # letters only (handles æøå)

def extract_words(s: str):
    # "words" = sequences of letters only; punctuation/hyphens/spaces ignored
    return WORD_RE.findall(unicodedata.normalize("NFC", s or ""))

def is_small_word_edit(a: str, b: str) -> bool:
    """
    Allows spelling/case tweaks, rejects real word substitutions.
    """
    a0 = (a or "").lower()
    b0 = (b or "").lower()
    if a0 == b0:
        return True

    # Very short words: be strict
    maxlen = max(len(a0), len(b0))
    if maxlen <= 3:
        return difflib.SequenceMatcher(a=a0, b=b0).ratio() >= 0.90

    # Normal words: allow typical typos (profesjonel->profesjonell etc.)
    ratio = difflib.SequenceMatcher(a=a0, b=b0).ratio()
    return ratio >= 0.70

def violates_no_word_add_remove(original: str, corrected: str) -> bool:
    """
    True if model added/removed/replaced whole words (not just spelling).
    """
    ow = extract_words(original)
    cw = extract_words(corrected)

    # Added/removed words
    if len(ow) != len(cw):
        return True

    # Word-by-word substitution (synonyms / rewrites)
    for a, b in zip(ow, cw):
        if not is_small_word_edit(a, b):
            return True

    return False



def project_safe_word_corrections(original: str, corrected: str) -> str:
    """
    Salvage mode:
    - Never adds/removes/reorders words
    - Applies ONLY 1-to-1 small spelling edits to existing words in the original text
    - Preserves original whitespace/punctuation exactly
    """
    orig = unicodedata.normalize("NFC", original or "")
    corr = unicodedata.normalize("NFC", corrected or "")

    orig_matches = list(WORD_RE.finditer(orig))
    corr_words = extract_words(corr)

    orig_words = [m.group(0) for m in orig_matches]
    if not orig_words or not corr_words:
        return original

    sm = difflib.SequenceMatcher(
        a=[w.lower() for w in orig_words],
        b=[w.lower() for w in corr_words],
        autojunk=False
    )

    # Collect replacements as (start, end, new_word)
    reps = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        if (i2 - i1) == 1 and (j2 - j1) == 1:
            a = orig_words[i1]
            b = corr_words[j1]
            if is_small_word_edit(a, b):  # spelling-level only
                m = orig_matches[i1]
                reps.append((m.start(), m.end(), b))

    if not reps:
        return original

    # Apply from end → start so offsets don't shift
    out = orig
    for s, e, nw in sorted(reps, key=lambda x: x[0], reverse=True):
        out = out[:s] + nw + out[e:]

    return out


# =================================================
# OPENAI CLIENT
# =================================================

# Uses OPENAI_API_KEY from environment (systemd/gunicorn env or .env you load elsewhere)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =================================================
# MAIN VIEW
# =================================================

def index(request):
    # AJAX POST → return JSON with corrected text + diffs
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        text = request.POST.get("text", "")

        if not text.strip():
            return JsonResponse({
                "original_text": "",
                "corrected_text": "",
                "differences": [],
                "error_count": 0,
            })

        corrected_text = correct_with_openai(text)
        differences = find_differences_charwise(text, corrected_text)

        return JsonResponse({
            "original_text": text,
            "corrected_text": corrected_text,
            "differences": differences,
            "error_count": len(differences),
        })

    # Normal GET (and any non-AJAX POST fallback)
    return render(request, "checker/index.html")


# =================================================
# OPENAI – NORWEGIAN (MIRRORS DANISH STYLE)
# =================================================
def correct_with_openai(text: str) -> str:
    """
    Hard constraints:
    - never add/remove/reorder words
    - allow spelling + punctuation + spacing
    - we also undo pure word-merges like "alt for" -> "altfor"
    """
    try:
        base_prompt = (
            "Du er en profesjonell norsk korrekturleser.\n\n"
            "VIKTIGE REGLER (MÅ FØLGES):\n"
            "- IKKE legg til nye ord\n"
            "- IKKE fjern ord\n"
            "- IKKE bytt ut ord med andre ord (ingen synonymer)\n"
            "- IKKE endre rekkefølgen på ord\n"
            "- Du kan rette STAVEFEIL inne i eksisterende ord, og rette tegnsetting/komma/store bokstaver/mellomrom\n"
            "- Hvis en endring krever at et ord må legges til/fjernes, LA DET STÅ\n\n"
            "Returner KUN den korrigerte teksten uten forklaring."
        )

        def call_llm(system_prompt: str, user_text: str) -> str:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()

        # 1) First attempt
        corrected = call_llm(base_prompt, text)
        if not corrected:
            return text

        corrected = undo_space_merges(text, corrected)

        # 2) If unchanged, retry once with a nudge (THIS is what you lost before)
        if corrected.strip() == text.strip():
            nudge_prompt = base_prompt + (
                "\n\nTEKSTEN INNEHOLDER FEIL.\n"
                "Du må rette alle tydelige stavefeil og tegnsettingsfeil innenfor reglene.\n"
                "Ikke returner identisk tekst hvis det finnes feil."
            )
            corrected2 = call_llm(nudge_prompt, text)
            if corrected2:
                corrected2 = undo_space_merges(text, corrected2)
                corrected = corrected2

        # 3) Validate: if model added/removed/substituted whole words → retry strict once
        if violates_no_word_add_remove(text, corrected):
            strict_prompt = base_prompt + (
                "\n\nEKSTRA STRIKT:\n"
                "- Antall ord i svaret MÅ være identisk med input\n"
                "- Hvert ord i output skal være samme ord som input (kun små staveendringer er lov)\n"
                "- Ikke forbedre setninger eller flyt; kun rett skrivefeil og tegnsetting.\n"
            )
            corrected2 = call_llm(strict_prompt, text)
            if corrected2:
                corrected2 = undo_space_merges(text, corrected2)
                if not violates_no_word_add_remove(text, corrected2):
                    return corrected2

            # 4) Salvage instead of returning original:
            # apply only safe spelling fixes to existing words (keeps word count/order)
            salvaged = project_safe_word_corrections(text, corrected2 or corrected)
            return salvaged if salvaged else text

        return corrected

    except Exception as e:
        print("❌ OpenAI error:", e)
        return text


WS_TOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)

def undo_space_merges(original: str, corrected: str, max_merge_words: int = 3) -> str:
    """
    Reverts corrections that ONLY merge multiple letter-words by removing spaces:
      "alt for" -> "altfor"
      "privat livet" -> "privatlivet"

    It does NOT touch hyphenations like:
      "e - poster" -> "e-poster"
    because that's not a pure space-removal merge.

    Keeps original whitespace between the words (so line breaks stay line breaks).
    """
    if not original or not corrected:
        return corrected

    orig_full = WS_TOKEN_RE.findall(unicodedata.normalize("NFC", original))
    corr_full = WS_TOKEN_RE.findall(unicodedata.normalize("NFC", corrected))

    def is_ws(t: str) -> bool:
        return t.isspace()

    # Letters-only (no digits/underscore). Works for Norwegian letters too.
    def is_word(t: str) -> bool:
        return bool(re.fullmatch(r"[^\W\d_]+", t, re.UNICODE))

    # Build "significant token" lists (no whitespace) + map sig-index -> full-index
    orig_sig, orig_map = [], []
    for idx, tok in enumerate(orig_full):
        if not is_ws(tok):
            orig_sig.append(tok)
            orig_map.append(idx)

    corr_sig, corr_map = [], []
    for idx, tok in enumerate(corr_full):
        if not is_ws(tok):
            corr_sig.append(tok)
            corr_map.append(idx)

    # Lowercase for matching
    sm = difflib.SequenceMatcher(
        a=[t.lower() for t in orig_sig],
        b=[t.lower() for t in corr_sig],
        autojunk=False,
    )

    replacements = {}  # corr_full_index -> replacement_string

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue

        # We only care about N words -> 1 word
        if (j2 - j1) != 1:
            continue
        n = (i2 - i1)
        if not (2 <= n <= max_merge_words):
            continue

        corr_tok = corr_sig[j1]
        if not is_word(corr_tok):
            continue
        if not all(is_word(t) for t in orig_sig[i1:i2]):
            continue

        # Pure merge check: join original words equals corrected word (case-insensitive)
        if "".join(orig_sig[i1:i2]).lower() != corr_tok.lower():
            continue

        # Make sure the original region between these word tokens contains ONLY whitespace
        start_full = orig_map[i1]
        end_full = orig_map[i2 - 1]
        between = orig_full[start_full:end_full + 1]
        if sum(1 for t in between if not is_ws(t)) != n:
            continue

        replacement_str = "".join(between)  # preserves original whitespace between words
        corr_full_index = corr_map[j1]
        replacements[corr_full_index] = replacement_str

    if not replacements:
        return corrected

    # Apply replacements (no index shifting; we replace token content only)
    for idx, rep in replacements.items():
        corr_full[idx] = rep

    return "".join(corr_full)


# =================================================
# DIFF ENGINE (IDENTICAL TO DANISH)
# =================================================
def find_differences_charwise(original: str, corrected: str):
    """
    Robust token diff that:
    - handles merges/splits (e.g., 'alt for' -> 'altfor', 'e - poster' -> 'e-poster')
    - returns original-string char spans (start/end) so frontend can highlight precisely
    - groups adjacent diffs into larger 'areas' to avoid highlighting every single word
    """
    orig_text = unicodedata.normalize("NFC", (original or "").replace("\r\n", "\n").replace("\r", "\n"))
    corr_text = unicodedata.normalize("NFC", (corrected or "").replace("\r\n", "\n").replace("\r", "\n"))

    if not orig_text and not corr_text:
        return []

    token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokens_with_spans(s: str):
        toks, spans = [], []
        for m in token_re.finditer(s):
            toks.append(m.group(0))
            spans.append((m.start(), m.end()))
        return toks, spans

    orig_tokens, orig_spans = tokens_with_spans(orig_text)
    corr_tokens, corr_spans = tokens_with_spans(corr_text)

    def span_for_token_range(spans, i1, i2, text_len):
        """Char span from first token start to last token end, including any whitespace between."""
        if not spans:
            return 0, 0
        if i1 >= len(spans):
            return text_len, text_len
        if i1 == i2:
            # insertion point: before token i1
            return spans[i1][0], spans[i1][0]
        return spans[i1][0], spans[i2 - 1][1]

    def norm_no_space(s: str) -> str:
        # remove whitespace only; keep punctuation so 'e - poster' ~ 'e-poster'
        return re.sub(r"\s+", "", s.lower())

    def similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(a=a, b=b).ratio()

    def is_pure_punct(s: str) -> bool:
        # punctuation-only string (commas, periods, hyphens, etc.)
        return bool(re.fullmatch(r"[^\w\s]+", s, re.UNICODE))

    # Important: disable autojunk (it can behave oddly on short/repetitive text)
    sm = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens, autojunk=False)

    raw_diffs = []

    # Build raw diffs from opcodes
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        o_start, o_end = span_for_token_range(orig_spans, i1, i2, len(orig_text))
        c_start, c_end = span_for_token_range(corr_spans, j1, j2, len(corr_text))

        o_chunk = orig_text[o_start:o_end]
        c_chunk = corr_text[c_start:c_end]

        # Keep things local (prevents huge “rewrite” highlights)
        o_tok_count = i2 - i1
        c_tok_count = j2 - j1
        if (o_tok_count + c_tok_count) > 14:
            continue
        if (len(o_chunk) + len(c_chunk)) > 180:
            continue

        if tag == "replace":
            # Accept if it's basically a local correction OR a whitespace-merge/split
            # (alt for -> altfor, e - poster -> e-poster, privat livet -> privatlivet)
            if norm_no_space(o_chunk) == norm_no_space(c_chunk) or similarity(o_chunk.lower(), c_chunk.lower()) >= 0.55:
                raw_diffs.append({
                    "type": "replace",
                    "start": o_start,
                    "end": o_end,
                    "original": o_chunk,
                    "suggestion": c_chunk,
                    # extra fields (safe if frontend ignores them)
                    "c_start": c_start,
                    "c_end": c_end,
                })

        elif tag == "insert":
            # Only surface punctuation inserts (commas/periods/etc.) to avoid spacing issues
            if c_chunk and is_pure_punct(c_chunk):
                raw_diffs.append({
                    "type": "insert",
                    "start": o_start,
                    "end": o_start,
                    "original": "",
                    "suggestion": c_chunk,
                    "c_start": c_start,
                    "c_end": c_end,
                })

        elif tag == "delete":
            # Only surface small deletes (usually punctuation / tiny tokens)
            if o_chunk and (is_pure_punct(o_chunk) or len(o_chunk.strip()) <= 2):
                raw_diffs.append({
                    "type": "delete",
                    "start": o_start,
                    "end": o_end,
                    "original": o_chunk,
                    "suggestion": "",
                    "c_start": c_start,
                    "c_end": c_end,
                })

    if not raw_diffs:
        return []

    # Sort and GROUP into “areas” (merge diffs separated only by whitespace)
    raw_diffs.sort(key=lambda d: (d["start"], d["end"]))

    grouped = [raw_diffs[0]]
    for d in raw_diffs[1:]:
        prev = grouped[-1]

        gap = orig_text[prev["end"]:d["start"]]
        gap_is_only_ws = (gap.strip() == "")

        # Merge if edits are basically adjacent (only spaces/newlines between)
        if gap_is_only_ws and (d["start"] <= prev["end"] + 2):
            prev["end"] = max(prev["end"], d["end"])
            prev["start"] = min(prev["start"], d["start"])

            # Rebuild the displayed chunks (covers merges like "alt for" cleanly)
            prev["original"] = orig_text[prev["start"]:prev["end"]]

            # Best-effort: if we have corrected spans, merge them too
            if "c_start" in prev and "c_start" in d:
                prev["c_start"] = min(prev["c_start"], d["c_start"])
                prev["c_end"] = max(prev["c_end"], d["c_end"])
                prev["suggestion"] = corr_text[prev["c_start"]:prev["c_end"]]
            else:
                # fallback: keep previous suggestion
                pass

            prev["type"] = "replace"
        else:
            grouped.append(d)

    # Optional: dedupe identical spans
    out = []
    seen = set()
    for d in grouped:
        key = (d["start"], d["end"], d.get("suggestion", ""))
        if key in seen:
            continue
        seen.add(key)
        # Keep only what your frontend expects (extras are okay to keep too)
        out.append({
            "type": d["type"],
            "start": d["start"],
            "end": d["end"],
            "original": d["original"],
            "suggestion": d["suggestion"],
        })

    return out


# =================================================
# AUTH (UNCHANGED, KEPT MINIMAL)
# =================================================

from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.models import User


def register(request):
    if request.method != "POST":
        return redirect("index")

    email = request.POST.get("email")
    password = request.POST.get("password")
    name = request.POST.get("name")

    if User.objects.filter(username=email).exists():
        messages.error(request, "E-post finnes allerede.")
        return redirect("/")

    user = User.objects.create_user(
        username=email,
        email=email,
        password=password,
        first_name=name,
    )

    login(request, user)
    return redirect("/")


def login_view(request):
    if request.method != "POST":
        return redirect("/")

    user = authenticate(
        request,
        username=request.POST.get("email"),
        password=request.POST.get("password"),
    )

    if user is None:
        messages.error(request, "Feil e-post eller passord.")
        return redirect("/")

    login(request, user)
    return redirect("/")


def logout_view(request):
    if request.method == "POST":
        logout(request)
    return redirect("/")
