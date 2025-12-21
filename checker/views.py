from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import logging
import re
import difflib
import unicodedata

client = OpenAI()
logger = logging.getLogger(__name__)

# -----------------------------
# Charwise diff (same style as Danish)
# -----------------------------

def find_differences_charwise(original: str, corrected: str):
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

    orig_positions = []
    cursor = 0
    for tok in orig_tokens:
        start_idx = orig_text.find(tok, cursor)
        end_idx = start_idx + len(tok)
        orig_positions.append((start_idx, end_idx))
        cursor = end_idx

    def span_for_range(i_start, i_end_exclusive):
        if i_start >= len(orig_positions):
            return len(orig_text), len(orig_text)
        if i_start == i_end_exclusive:
            start_i, _ = orig_positions[i_start]
            return start_i, start_i
        return orig_positions[i_start][0], orig_positions[i_end_exclusive - 1][1]

    def tokens_are_small_edit(a, b):
        if a.lower().strip(",.;:!?") == b.lower().strip(",.;:!?"):
            return True
        return difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio() >= 0.6

    def is_pure_punctuation(tok):
        return bool(re.fullmatch(r"[,.:;!?]+", tok))

    sm = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        if tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
            o, c = orig_tokens[i1], corr_tokens[j1]
            if tokens_are_small_edit(o, c):
                s, e = span_for_range(i1, i2)
                diffs_out.append({
                    "type": "replace",
                    "start": s,
                    "end": e,
                    "original": o,
                    "suggestion": c,
                })

        elif tag == "delete" and (i2 - i1) == 1:
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

        elif tag == "insert" and (j2 - j1) == 1:
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


# -----------------------------
# OpenAI correction (NOW same logic as Danish)
# -----------------------------

def correct_with_openai_no(text: str) -> str:
    """
    Produces a fully corrected Norwegian (Bokmål) version of the input text:
    - Fixes spelling, grammar, punctuation, capitalization, and comma errors
    - Keeps tone and meaning identical
    - Avoids stylistic rewriting or added exclamation marks
    Uses the SAME retry pattern as the Danish version.
    """
    try:
        base_prompt = (
            "Du er en profesjonell norsk språkvasker. "
            "Din oppgave er å returnere teksten i en PERFEKT, grammatisk korrekt, "
            "og naturlig form på norsk bokmål. "
            "Du skal rette ALLE feil i stavning, bøyning, grammatikk, ordstilling, "
            "store bokstaver, mellomrom, og tegnsetting, og særlig kommatering. "
            "Ret også komma etter reglene for startkomma (komma før leddsetninger) "
            "samt komma mellom sideordnede setninger. "
            "Behold tekstens betydning, stil og tone uendret. "
            "Ikke legg til ekstra ord, tegn eller utropstegn. "
            "Returner KUN den korrigerte teksten uten noen forklaring."
        )

        # First attempt
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        corrected = (resp.choices[0].message.content or "").strip()

        # If the text barely changed, retry with stronger emphasis (same pattern as Danish)
        if corrected.strip() == text.strip():
            print("⚠ No change detected – retrying with stricter correction mode...")
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": base_prompt
                        + " Hvis du er i tvil, så rett heller for mye enn for lite. "
                          "Finn ALLE feil, også små komma- og formuleringsfeil."
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            corrected2 = (resp2.choices[0].message.content or "").strip()
            if corrected2:
                corrected = corrected2

        return corrected if corrected else text

    except Exception:
        logger.exception("OpenAI error")
        return text


# -----------------------------
# Main view
# -----------------------------

def index(request):
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        original_text = request.POST.get("text", "")

        if not original_text.strip():
            return JsonResponse({
                "original_text": "",
                "corrected_text": "",
                "differences": [],
                "error_count": 0,
            })

        corrected = correct_with_openai_no(original_text)
        differences = find_differences_charwise(original_text, corrected)

        return JsonResponse({
            "original_text": original_text,
            "corrected_text": corrected,
            "differences": differences,
            "error_count": len(differences),
        })

    return render(request, "checker/index.html")


# -----------------------------
# Auth (unchanged)
# -----------------------------

from django.contrib.auth.models import User
from django.contrib.auth import login
from django.contrib import messages

def register(request):
    if request.method != "POST":
        return redirect("index")

    name = request.POST.get("name")
    email = request.POST.get("email")
    password = request.POST.get("password")

    if User.objects.filter(username=email).exists():
        messages.error(request, "E-postadressen används redan.")
        return redirect(request.POST.get("next", "/"))

    user = User.objects.create_user(
        username=email,
        email=email,
        password=password,
        first_name=name,
    )

    login(request, user)
    return redirect(request.POST.get("next", "/"))


from django.contrib.auth import authenticate

def login_view(request):
    if request.method != "POST":
        return redirect("/")

    email = request.POST.get("email")
    password = request.POST.get("password")

    if not email or not password:
        messages.error(request, "Fyll i både e-post och lösenord.")
        return redirect(request.POST.get("next", "/"))

    user = authenticate(request, username=email, password=password)

    if user is None:
        messages.error(request, "Fel e-post eller lösenord.")
        return redirect(request.POST.get("next", "/"))

    login(request, user)
    return redirect(request.POST.get("next", "/"))


from django.contrib.auth import logout

def logout_view(request):
    if request.method == "POST":
        logout(request)
    return redirect("/")
