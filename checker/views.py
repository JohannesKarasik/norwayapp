from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import os
import re
import difflib
import unicodedata


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
    Fully corrects Norwegian text:
    - Fixes spelling, grammar, punctuation, capitalization, and commas
    - Keeps meaning/tone
    - Avoids stylistic rewriting and extra punctuation
    """
    try:
        base_prompt = (
            "Du er en profesjonell norsk språkvasker. "
            "Din oppgave er å returnere teksten i en PERFEKT, grammatisk korrekt "
            "og naturlig form på norsk. "
            "Du skal rette ALLE feil i rettskrivning, bøyning, ordstilling, "
            "store bokstaver, mellomrom og tegnsetting. "
            "Behold tekstens betydning, stil og tone uendret. "
            "Legg aldri til ekstra ord, tegn eller utropstegn. "
            "Returner KUN den korrigerte teksten uten forklaring."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        corrected = (resp.choices[0].message.content or "").strip()

        # Retry once if model returned unchanged text
        if corrected.strip() == text.strip():
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": base_prompt
                        + " Hvis du er i tvil, rett heller for mye enn for lite. "
                          "Finn ALLE feil, også små rettskrivnings- og tegnsettingsfeil."
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            corrected2 = (resp2.choices[0].message.content or "").strip()
            if corrected2:
                corrected = corrected2

        return corrected if corrected else text

    except Exception as e:
        print("❌ OpenAI error:", e)
        return text


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
