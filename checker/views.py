from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import logging
import re
import difflib

client = OpenAI()
logger = logging.getLogger(__name__)

# -----------------------------
# Tokenization (alignment-safe)
# -----------------------------

TOKEN_RE = re.compile(r"\S+")

def tokenize_with_offsets(text: str):
    """
    Returns list of (token, start, end)
    preserving exact character offsets.
    """
    return [
        (m.group(), m.start(), m.end())
        for m in TOKEN_RE.finditer(text)
    ]


# -----------------------------
# OpenAI correction (SAFE MODE)
# -----------------------------

def correct_with_openai_no(text: str) -> str:
    """
    Norwegian spellcheck ONLY.
    No word order changes, no spacing changes,
    no punctuation movement.
    """
    try:
        system_prompt = (
            "Du er en ekstremt streng norsk språkkorrektør.\n\n"

            "ABSOLUTTE REGLER (MÅ FØLGES):\n"
            "- IKKE legg til ord\n"
            "- IKKE fjern ord\n"
            "- IKKE endre rekkefølgen på ord\n"
            "- IKKE del eller slå sammen ord\n"
            "- IKKE flytt, legg til eller fjern tegnsetting\n"
            "- IKKE endre mellomrom eller linjeskift\n\n"

            "DU HAR KUN LOV TIL Å:\n"
            "- rette rene stavefeil inne i samme ord\n"
            "- rette små bøyningsfeil av samme ord\n\n"

            "HVIS DU ER I TVIL → IKKE ENDRE.\n\n"
            "Returner KUN teksten, uten forklaring."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )

        corrected = (resp.choices[0].message.content or "")

        # HARD SAFETY: token count must match
        if len(tokenize_with_offsets(text)) != len(tokenize_with_offsets(corrected)):
            logger.warning("Token count mismatch – disabling correction")
            return text

        return corrected if corrected else text

    except Exception:
        logger.exception("OpenAI error")
        return text


# -----------------------------
# Token-wise diff (bulletproof)
# -----------------------------

def find_differences_tokenwise(original: str, corrected: str):
    diffs = []

    orig_tokens = tokenize_with_offsets(original)
    corr_tokens = tokenize_with_offsets(corrected)

    # Abort immediately if alignment breaks
    if len(orig_tokens) != len(corr_tokens):
        return []

    for (o_tok, o_start, o_end), (c_tok, _, _) in zip(orig_tokens, corr_tokens):
        if o_tok == c_tok:
            continue

        o_core = o_tok.lower().strip(".,;:!?")
        c_core = c_tok.lower().strip(".,;:!?")

        # Allow only small spelling corrections
        if difflib.SequenceMatcher(a=o_core, b=c_core).ratio() < 0.85:
            return []  # abort entire diff safely

        diffs.append({
            "type": "replace",
            "start": o_start,
            "end": o_end,
            "original": o_tok,
            "suggestion": c_tok,
        })

    return diffs


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
        differences = find_differences_tokenwise(original_text, corrected)

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
