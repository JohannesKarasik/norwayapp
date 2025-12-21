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
    Token-level diff with alignment.
    Highlights small local changes, ignores rewrites/reorders.
    """
    diffs_out = []

    orig_text = unicodedata.normalize("NFC", original)
    corr_text = unicodedata.normalize("NFC", corrected)

    def merge_punctuation(tokens):
        merged = []
        for tok in tokens:
            if merged and re.match(r"^[,.:;!?]$", tok):
                merged[-1] += tok
            else:
                merged.append(tok)
        return merged

    orig_tokens = merge_punctuation(re.findall(r"\w+|[^\w\s]", orig_text, re.UNICODE))
    corr_tokens = merge_punctuation(re.findall(r"\w+|[^\w\s]", corr_text, re.UNICODE))

    # Map original tokens to char spans in original string
    orig_positions = []
    cursor = 0
    for tok in orig_tokens:
        start = orig_text.find(tok, cursor)
        if start == -1:
            return []
        end = start + len(tok)
        orig_positions.append((start, end))
        cursor = end

    def span_for_range(i_start, i_end):
        if i_start >= len(orig_positions):
            return len(orig_text), len(orig_text)
        if i_start == i_end:
            s, _ = orig_positions[i_start]
            return s, s
        return orig_positions[i_start][0], orig_positions[i_end - 1][1]

    def tokens_are_small_edit(a, b):
        a_low = a.lower()
        b_low = b.lower()

        # Case/punctuation-only change
        if a_low.strip(",.;:!?") == b_low.strip(",.;:!?"):
            return True

        ratio = difflib.SequenceMatcher(a=a_low, b=b_low).ratio()
        return ratio >= 0.6

    def is_pure_punctuation(tok):
        return bool(re.fullmatch(r"[,.:;!?]+", tok))

    sm = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        if tag == "replace":
            # Only accept 1-to-1 small edits
            if (i2 - i1) == 1 and (j2 - j1) == 1:
                o = orig_tokens[i1]
                c = corr_tokens[j1]
                if tokens_are_small_edit(o, c):
                    s, e = span_for_range(i1, i2)
                    diffs_out.append({
                        "type": "replace",
                        "start": s,
                        "end": e,
                        "original": o,
                        "suggestion": c,
                    })

        elif tag == "delete":
            # Only show small deletes (punctuation or tiny tokens)
            if (i2 - i1) == 1:
                o = orig_tokens[i1]
                if is_pure_punctuation(o) or len(o) <= 2:
                    s, e = span_for_range(i1, i2)
                    diffs_out.append({
                        "type": "delete",
                        "start": s,
                        "end": e,
                        "original": o,
                        "suggestion": "",
                    })

        elif tag == "insert":
            # Only show punctuation inserts
            if (j2 - j1) == 1:
                c = corr_tokens[j1]
                if is_pure_punctuation(c):
                    s, _ = span_for_range(i1, i1)
                    diffs_out.append({
                        "type": "insert",
                        "start": s,
                        "end": s,
                        "original": "",
                        "suggestion": c,
                    })

    return diffs_out


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
