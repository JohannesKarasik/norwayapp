from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import os
import re
import difflib
import unicodedata



from openai import OpenAI
import os

_openai_client = None

def get_openai_client():
    global _openai_client

    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        _openai_client = OpenAI(api_key=api_key)

    return _openai_client


# --- Paste normalization (Google Docs, Word, etc.) ---
ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")  # ZWSP/ZWNJ/ZWJ/WJ/BOM

def normalize_pasted_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)

    # Normalize all common "line break" variants to \n
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")  # Unicode LS/PS
    s = s.replace("\u000b", "\n").replace("\u000c", "\n")  # VT/FF

    # Normalize non-breaking spaces to normal spaces
    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2007", " ")

    # Remove invisible zero-width characters (they break matching/diffs)
    s = ZERO_WIDTH_RE.sub("", s)

    return s


WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # letters only (handles æøå)

def extract_words(s: str):
    # "words" = sequences of letters only; punctuation/hyphens/spaces ignored
    return WORD_RE.findall(unicodedata.normalize("NFC", s or ""))

def edit_distance_leq1(a: str, b: str) -> bool:
    """
    True if Levenshtein distance <= 1 (fast path).
    Allows 1 insert/delete/replace.
    """
    a = (a or "")
    b = (b or "")
    if a == b:
        return True

    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    # Same length: at most 1 substitution
    if la == lb:
        mismatches = sum(1 for x, y in zip(a, b) if x != y)
        return mismatches <= 1

    # Ensure a is the shorter
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    # lb = la + 1: at most 1 insertion
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1  # skip one char in longer string
    return True


# Allow-list for common Norwegian confusions that are NOT "synonyms"
ALLOWED_WORD_SWAPS = {
    "de": {"dem"},
    "dem": {"de"},
    "en": {"enn"},
    "enn": {"en"},
    # add more if you want later:
    # "og": {"å"}, "å": {"og"}  # (optional, only if you want to allow this)
}

def is_small_word_edit(a: str, b: str) -> bool:
    """
    Allows:
    - spelling tweaks (1–2 char edits)
    - short grammar swaps like de/dem, en/enn (allow-list)
    Rejects:
    - real rewrites/synonyms (low similarity / large changes)
    """
    a0 = (a or "").lower()
    b0 = (b or "").lower()

    if a0 == b0:
        return True

    if a0 in ALLOWED_WORD_SWAPS and b0 in ALLOWED_WORD_SWAPS[a0]:
        return True

    maxlen = max(len(a0), len(b0))

    # For short words, use edit distance (SequenceMatcher ratio is misleading here)
    if maxlen <= 4:
        return edit_distance_leq1(a0, b0)

    # For normal words, allow typical misspellings
    ratio = difflib.SequenceMatcher(a=a0, b=b0).ratio()
    return ratio >= 0.72


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



# =================================================
# MAIN VIEW
# =================================================


def chunk_text_preserve(text: str, max_chars: int = 1800):
    """
    Split tekst i chunks (bevarer whitespace), så lange tekster ikke kollapser til 0 diffs.
    Primært på sætninger/linjeskift, fallback til hård split.
    """
    if not text:
        return [""]

    # Split på sætninger + store linjeskift, men behold delimiters i output
    units = re.findall(r".*?(?:[.!?]+(?:\s+|$)|\n{2,}|$)", text, flags=re.S)
    units = [u for u in units if u]  # fjern tomme

    if not units:
        units = [text]

    chunks = []
    buf = ""
    for u in units:
        if len(buf) + len(u) <= max_chars:
            buf += u
        else:
            if buf:
                chunks.append(buf)
                buf = ""
            # Hvis en enkelt unit er for stor, split hårdt
            if len(u) > max_chars:
                for i in range(0, len(u), max_chars):
                    chunks.append(u[i:i + max_chars])
            else:
                buf = u

    if buf:
        chunks.append(buf)

    return chunks


def correct_with_openai_chunked(text: str, max_chars: int = 1800) -> str:
    """
    Kør korrektur i chunks for lange tekster.
    """
    parts = chunk_text_preserve(text, max_chars=max_chars)
    out = []
    for p in parts:
        if p.strip():
            out.append(correct_with_openai(p))
        else:
            out.append(p)
    return "".join(out)


from django.http import JsonResponse
from django.http import JsonResponse
from django.shortcuts import render

def index(request):
    # =========================
    # AJAX TEXT CORRECTION
    # =========================
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        if not request.user.is_authenticated:
            return JsonResponse({"error": "auth_required"}, status=401)

        profile = getattr(request.user, "profile", None)
        if not profile or not profile.is_paying:
            return JsonResponse({"error": "payment_required"}, status=402)

        text = normalize_pasted_text(request.POST.get("text", ""))

        if not text.strip():
            return JsonResponse({
                "original_text": "",
                "corrected_text": "",
                "differences": [],
                "error_count": 0,
            })

        # ✅ Chunk correction når teksten er lang
        if len(text) > 2000:
            corrected_text = correct_with_openai_chunked(text, max_chars=1800)
        else:
            corrected_text = correct_with_openai(text)

        # Normal diff (stram)
        differences = find_differences_charwise(text, corrected_text)

        # ✅ Hvis der ER ændringer men 0 diffs (typisk ved lange tekster / mange kommaer)
        if not differences and corrected_text.strip() != text.strip():
            differences = find_differences_charwise(
                text,
                corrected_text,
                max_block_tokens=80,
                max_block_chars=1200,
                max_diffs=300,
            )

        return JsonResponse({
            "original_text": text,
            "corrected_text": corrected_text,
            "differences": differences,
            "error_count": len(differences),
        })

    # =========================
    # PAGE RENDER (IMPORTANT)
    # =========================
    is_paying = (
        request.user.is_authenticated
        and hasattr(request.user, "profile")
        and request.user.profile.is_paying
    )

    return render(
        request,
        "checker/index.html",
        {
            "is_paying": is_paying,
        }
    )


def same_words_exact(a: str, b: str) -> bool:
    # Komma-pass må IKKE ændre bogstaver/ord — kun komma/whitespace.
    return extract_words(a) == extract_words(b)

ONLY_COMMA_WS_RE = re.compile(r"^[\s,]*$", re.UNICODE)

def _adjacent_has_comma(orig: str, i1: int, i2: int, cand: str, j1: int, j2: int) -> bool:
    def ch(s: str, idx: int) -> str:
        return s[idx] if 0 <= idx < len(s) else ""
    # Check character right before/after the edited segment in either string
    return (
        ch(orig, i1 - 1) == "," or ch(orig, i2) == "," or
        ch(cand, j1 - 1) == "," or ch(cand, j2) == ","
    )

def keep_only_comma_changes(original: str, candidate: str) -> str:
    """
    Keep ONLY:
    - comma insert/remove
    - whitespace changes that are directly adjacent to a comma (space after comma etc.)
    Reject whitespace changes elsewhere (prevents merges like 'privat livet' -> 'privatlivet').
    """
    orig = unicodedata.normalize("NFC", original or "")
    cand = unicodedata.normalize("NFC", candidate or "")

    if not orig or not cand:
        return original

    sm = difflib.SequenceMatcher(a=orig, b=cand, autojunk=False)
    out = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(orig[i1:i2])
            continue

        oseg = orig[i1:i2]
        cseg = cand[j1:j2]

        # Only allow edits that are commas/whitespace
        if ONLY_COMMA_WS_RE.fullmatch(oseg) and ONLY_COMMA_WS_RE.fullmatch(cseg):
            # Always allow if a comma is involved, or the edit touches a comma
            if ("," in oseg) or ("," in cseg) or _adjacent_has_comma(orig, i1, i2, cand, j1, j2):
                out.append(cseg)
            else:
                # Disallow whitespace-only edits away from commas (prevents word merges/splits)
                out.append(oseg)
        else:
            # Revert anything else (word changes, hyphens, etc.)
            out.append(oseg)

    return "".join(out)



def insert_commas_with_openai(text: str) -> str:
    """
    Inserts/removes commas ONLY.
    Must not change words/letters/case. Only commas + whitespace around commas.
    """
    try:
        system_prompt = (
            "Du er en norsk komma-korrektør.\n\n"
            "REGLER (MÅ FØLGES):\n"
            "- Du får en tekst og du skal KUN rette komma.\n"
            "- IKKE ændr eller ret STAVNING, store/små bogstaver eller ordvalg.\n"
            "- IKKE tilføj eller fjern ord.\n"
            "- IKKE ændr rækkefølgen på ord.\n"
            "- Du kan KUN sette inn/fjerne komma.\n"
            "- Du kan KUN endre mellomrom rett før eller rett etter et komma.\n"
            "- Du må ALDRI fjerne/legge til mellomrom mellom to ord (ikke slå sammen eller splitte ord).\n\n"
            "Returner KUN teksten."
        )

        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").rstrip(" \t")

        if not out:
            return text

                # Hard validate: words must be EXACTLY identical
        # Keep only comma + whitespace changes (prevents eposter->e-poster etc. from nuking commas)
        safe = keep_only_comma_changes(text, out)
        safe = undo_space_merges(text, safe)  # extra safety if model still tries to merge words
        return safe



    except Exception as e:
        print("❌ OpenAI comma-only error:", e)
        return text


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
            "Du er en profesjonell norsk korrekturleser (bokmål).\n\n"
            "MÅL: Rett ALLE stavefeil og ALL tegnsetting, spesielt komma, uten å endre innhold.\n\n"
            "ABSOLUTTE REGLER (MÅ FØLGES):\n"
            "- IKKE legg til nye ord\n"
            "- IKKE fjern ord\n"
            "- IKKE endre rekkefølgen på ord\n"
            "- IKKE omskriv setninger og IKKE bruk synonymer\n"
            "- Du kan kun endre bokstaver inni eksisterende ord for å rette stavefeil\n"
            "- Du kan rette tegnsetting (komma/punktum/kolon/anførselstegn) og mellomrom\n"
            "- Bevar linjeskift og avsnitt nøyaktig som i input\n\n"
            "KOMMA-SJEKK (MÅ GJØRES FØR DU SVARER): Gå setning for setning og rett komma når regelen krever det:\n"
            "1) Komma etter innledende leddsetning:\n"
            "   Hvis/Når/Da/Dersom/Selv om/Fordi/Siden/Mens/Etter at/Før/For at/Om ... ,\n"
            "2) Komma rundt innskutte leddsetninger/parentetiske innskudd.\n"
            "3) Komma før 'og/men/for/eller' når det binder sammen to helsetninger "
            "(begge har eget subjekt + verbal).\n"
            "4) Komma i oppramsing når det trengs for tydelighet.\n"
            "5) IKKE sett komma mellom subjekt og verbal i en enkel helsetning.\n\n"
            "VIKTIG: Teksten inneholder feil. Du skal finne og rette dem innenfor reglene.\n"
            "Ikke returner identisk tekst hvis det finnes kommafeil eller tydelige skrivefeil.\n\n"
            "Returner KUN den korrigerte teksten. Ingen forklaring."
        )



        def call_llm(system_prompt: str, user_text: str) -> str:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").rstrip(" \t")


        # 1) First attempt
        corrected = call_llm(base_prompt, text)
        if not corrected:
            return text

        corrected = undo_space_merges(text, corrected)

        # 2) If unchanged, retry once with a nudge (THIS is what you lost before)
        if corrected.strip() == text.strip():
            nudge_prompt = base_prompt + (
                "\n\nTEKSTEN INNEHOLDER FEIL.\n"
                "Du må rette alle tydelige stavefeil OG alle kommafeil innenfor reglene.\n"
                "Kjør KOMMA-SJEKKEN (punkt 1–5) setning for setning og ikke returner identisk tekst hvis noen komma mangler/er feil."
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
            if not salvaged:
                return text

            # ✅ also run comma-only on salvaged text (safe)
            salvaged = insert_commas_with_openai(salvaged)
            return salvaged


        # ✅ second pass: comma-only (won't change words)
        corrected = insert_commas_with_openai(corrected)
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
def find_differences_charwise(original: str, corrected: str, max_block_tokens: int = 14, max_block_chars: int = 180, max_diffs: int = 250):
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
        if (o_tok_count + c_tok_count) > max_block_tokens:
            continue
        if (len(o_chunk) + len(c_chunk)) > max_block_chars:
            continue


        if tag == "replace":
            # Accept if it's basically a local correction OR a whitespace-merge/split
            if norm_no_space(o_chunk) == norm_no_space(c_chunk) or similarity(o_chunk.lower(), c_chunk.lower()) >= 0.55:
                raw_diffs.append({
                    "type": "replace",
                    "start": o_start,
                    "end": o_end,
                    "original": o_chunk,
                    "suggestion": c_chunk,
                    "c_start": c_start,
                    "c_end": c_end,
                })

        elif tag == "insert":
            # Punctuation-only inserts (like a missing comma) often become "0-length" diffs,
            # which the frontend can't highlight/click. Convert them into a small REPLACE
            # around nearby tokens, e.g. "vanskelig men" -> "vanskelig, men".
            if c_chunk and is_pure_punct(c_chunk):

                # Expand original context to 1 token left + 1 token right (when possible)
                left_i = max(i1 - 1, 0)
                right_i = min(i1 + 1, len(orig_tokens))  # range end is exclusive

                # Expand corrected context similarly (token before + inserted punct + token after)
                left_j = max(j1 - 1, 0)
                right_j = min(j2 + 1, len(corr_tokens))

                o_start2, o_end2 = span_for_token_range(orig_spans, left_i, right_i, len(orig_text))
                c_start2, c_end2 = span_for_token_range(corr_spans, left_j, right_j, len(corr_text))

                o_chunk2 = orig_text[o_start2:o_end2]
                c_chunk2 = corr_text[c_start2:c_end2]

                # Keep things local (same safety limits)
                if ((right_i - left_i) + (right_j - left_j)) > max_block_tokens:
                    continue
                if (len(o_chunk2) + len(c_chunk2)) > max_block_chars:
                    continue

                if o_chunk2 != c_chunk2:
                    raw_diffs.append({
                        "type": "replace",
                        "start": o_start2,
                        "end": o_end2,
                        "original": o_chunk2,
                        "suggestion": c_chunk2,
                        "c_start": c_start2,
                        "c_end": c_end2,
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
        gap_has_parabreak = ("\n\n" in gap)

        # Merge kun hvis der ikke er paragrafskift imellem
        if gap_is_only_ws and (not gap_has_parabreak) and (d["start"] <= prev["end"] + 2):

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

    return out[:max_diffs]


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
        messages.error(
            request,
            "Denne kontoen finnes ikke, eller passordet er feil."
        )
        return redirect("/")

    login(request, user)
    return redirect("/")

def logout_view(request):
    if request.method == "POST":
        logout(request)
    return redirect("/")

# checker/views.py

import stripe
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

stripe.api_key = settings.STRIPE_SECRET_KEY


@csrf_exempt
@require_POST
@login_required
def create_checkout_session(request):
    """
    Creates a Stripe Checkout Session for a subscription with trial.
    Redirects user back to `/` on success.
    User access is unlocked via webhook (source of truth).
    """

    # Safety: make sure user has an email (Stripe prefers it)
    customer_email = request.user.email or None

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",

            # 🔑 THIS IS CRITICAL — used by webhook
            client_reference_id=request.user.id,

            customer_email=customer_email,

            line_items=[
                {
                    "price": settings.STRIPE_PRICE_ID,
                    "quantity": 1,
                }
            ],

            subscription_data={
                "trial_period_days": 30,
            },

            success_url=request.build_absolute_uri("/"),
            cancel_url=request.build_absolute_uri("/"),

            # ✅ Promo/coupon box removed
            allow_promotion_codes=False,
        )

        return JsonResponse({"url": session.url})

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=400
        )


from django.contrib.auth.models import User
from .models import Profile

@csrf_exempt
def stripe_webhook(request):
    payload = request.body
    sig_header = request.META.get("HTTP_STRIPE_SIGNATURE")

    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            settings.STRIPE_WEBHOOK_SECRET,
        )
    except Exception:
        return HttpResponse(status=400)

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]

        user_id = session.get("client_reference_id")
        if not user_id:
            return HttpResponse(status=200)

        try:
            user = User.objects.get(id=user_id)
            profile, _ = Profile.objects.get_or_create(user=user)
            profile.is_paying = True
            profile.stripe_customer_id = session.get("customer")
            profile.stripe_subscription_id = session.get("subscription")
            profile.save()

            profile.save()
        except User.DoesNotExist:
            pass

    return HttpResponse(status=200)



@login_required
def cancel_subscription(request):
    profile = getattr(request.user, "profile", None)

    if not profile or not profile.stripe_subscription_id:
        messages.error(request, "Fant ikke aktivt abonnement.")
        return redirect("settings")

    try:
        stripe.checkout.Session.expire  # noop import guard

        stripe.Subscription.delete(profile.stripe_subscription_id)

        profile.is_paying = False
        profile.stripe_subscription_id = None
        profile.save()

        messages.success(request, "Abonnementet er avsluttet.")
    except Exception as e:
        messages.error(request, "Kunne ikke avslutte abonnementet.")

    return redirect("settings")



from django.contrib.auth.decorators import login_required
from django.contrib import messages

@login_required
def settings_view(request):
    user = request.user
    profile = getattr(user, "profile", None)

    if request.method == "POST":
        # Change password
        current = request.POST.get("current_password")
        new = request.POST.get("new_password")

        if not user.check_password(current):
            messages.error(request, "Nåværende passord er feil.")
            return redirect("settings")

        if not new or len(new) < 8:
            messages.error(request, "Passordet må være minst 8 tegn.")
            return redirect("settings")

        user.set_password(new)
        user.save()

        messages.success(request, "Passordet er oppdatert.")
        return redirect("settings")

    return render(
        request,
        "checker/settings.html",
        {
            "email": user.email,
            "is_paying": bool(profile and profile.is_paying),
        }
    )


