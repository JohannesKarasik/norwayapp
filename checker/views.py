# checker/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import logging
import re
import difflib
import unicodedata

client = OpenAI()
logger = logging.getLogger(__name__)


def correct_with_openai_sv(text: str) -> str:
    """
    Returns a corrected version of the text where:
    - NO words are added or removed
    - Word order is identical
    - Only spelling and punctuation attached to a word may change
    """
    try:
        system_prompt = (
            "Du er en profesjonell norsk språkre­daktør.\n\n"
            "VIKTIGE REGLER (MÅ FØLGES):\n"
            "- IKKE legg til nye ord\n"
            "- IKKE fjern ord\n"
            "- IKKE endre rekkefølgen på ord\n"
            "- IKKE del eller slå sammen ord\n"
            "- IKKE endre mellomrom eller linjeskift\n\n"
            "Du har KUN lov til å:\n"
            "- rette stavefeil INNE I et eksisterende ord\n"
            "- legge til eller fjerne tegnsetting SOM EN DEL AV ORDET "
            "(f.eks. 'att' → 'att,')\n\n"
            "Hvis en feil krever omskriving, LA DEN STÅ URØRT.\n\n"
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

        corrected = (resp.choices[0].message.content or "").strip()

        # HARD SAFETY: word count must match exactly
        if len(corrected.split()) != len(text.split()):
            logger.warning("Word count mismatch – falling back to original text")
            return text

        return corrected if corrected else text

    except Exception:
        logger.exception("OpenAI error")
        return text


def find_differences_charwise(original: str, corrected: str):
    """
    Token-level diff where:
    - punctuation is part of the word token
    - ONLY 1-to-1 token replacements are allowed
    - no inserts / deletes / reorders are surfaced
    """

    diffs_out = []

    orig_text = unicodedata.normalize("NFC", original)
    corr_text = unicodedata.normalize("NFC", corrected)

    # Token = word WITH optional attached punctuation
    token_pattern = r"\w+[.,;:!?]?"

    orig_tokens = re.findall(token_pattern, orig_text, re.UNICODE)
    corr_tokens = re.findall(token_pattern, corr_text, re.UNICODE)

    # Absolute safety: token count must match
    if len(orig_tokens) != len(corr_tokens):
        return []

    # Map original tokens to char positions
    orig_positions = []
    cursor = 0
    for tok in orig_tokens:
        start = orig_text.find(tok, cursor)
        end = start + len(tok)
        orig_positions.append((start, end))
        cursor = end

    def is_small_edit(a: str, b: str) -> bool:
        a_core = a.lower().strip(".,;:!?")
        b_core = b.lower().strip(".,;:!?")

        if a_core == b_core:
            return True

        # allow small spelling fixes only
        ratio = difflib.SequenceMatcher(a=a_core, b=b_core).ratio()
        return ratio >= 0.8

    for i, (orig_tok, corr_tok) in enumerate(zip(orig_tokens, corr_tokens)):
        if orig_tok == corr_tok:
            continue

        if not is_small_edit(orig_tok, corr_tok):
            # ignore semantic rewrites
            continue

        start, end = orig_positions[i]

        diffs_out.append({
            "type": "replace",
            "start": start,
            "end": end,
            "original": orig_tok,
            "suggestion": corr_tok,
        })

    return diffs_out


def index(request):
    # Handle AJAX correction (allow anonymous users)
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        text = (request.POST.get("text") or "").strip()

        if not text:
            return JsonResponse({
                "original_text": "",
                "corrected_text": "",
                "differences": [],
                "error_count": 0,
            })

        corrected = correct_with_openai_sv(text)
        differences = find_differences_charwise(text, corrected)

        return JsonResponse({
            "original_text": text,
            "corrected_text": corrected,
            "differences": differences,
            "error_count": len(differences),
        })

    return render(request, "checker/index.html")


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


from django.contrib.auth import authenticate, login
from django.contrib import messages

def login_view(request):
    if request.method != "POST":
        return redirect("/")

    email = request.POST.get("email")
    password = request.POST.get("password")

    if not email or not password:
        messages.error(request, "Fyll i både e-post och lösenord.")
        return redirect(request.POST.get("next", "/"))

    user = authenticate(
        request,
        username=email,
        password=password
    )

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
