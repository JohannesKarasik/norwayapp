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
    try:
        system_prompt = (
            "Du er en profesjonell norsk språkre­daktør. "
            "Din oppgave er å korrigere ALLE feil i rettskriving, grammatikk, "
            "ordstilling og tegnsetting (spesielt komma). "
            "Behold betydningen nøyaktig. "
            "Returner KUN den korrigerte teksten."
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
        return corrected if corrected else text

    except Exception:
        logger.exception("OpenAI error")
        return text


def find_differences_charwise(original: str, corrected: str):
    """
    Token-level diff with alignment.
    Highlights small, local corrections (spelling, commas),
    ignores large rewrites / reordering.
    """

    diffs_out = []

    # Normalize unicode (æ/ø/å etc.)
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

    # Tokenize
    orig_tokens = merge_punctuation(re.findall(r"\w+|[^\w\s]", orig_text, re.UNICODE))
    corr_tokens = merge_punctuation(re.findall(r"\w+|[^\w\s]", corr_text, re.UNICODE))

    # Map original tokens to char positions
    orig_positions = []
    cursor = 0
    for tok in orig_tokens:
        start = orig_text.find(tok, cursor)
        end = start + len(tok)
        orig_positions.append((start, end))
        cursor = end

    def span_for_range(i_start, i_end):
        if i_start >= len(orig_positions):
            return len(orig_text), len(orig_text)
        if i_start == i_end:
            start, _ = orig_positions[i_start]
            return start, start
        start = orig_positions[i_start][0]
        end = orig_positions[i_end - 1][1]
        return start, end

    def tokens_are_small_edit(a, b):
        a_low = a.lower().strip(",.;:!?")
        b_low = b.lower().strip(",.;:!?")
        if a_low == b_low:
            return True
        return difflib.SequenceMatcher(a=a_low, b=b_low).ratio() >= 0.6

    def is_pure_punctuation(tok):
        return bool(re.fullmatch(r"[,.:;!?]+", tok))

    sm = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        if tag == "replace":
            if (i2 - i1) == 1 and (j2 - j1) == 1:
                orig_tok = orig_tokens[i1]
                corr_tok = corr_tokens[j1]
                if tokens_are_small_edit(orig_tok, corr_tok):
                    start, end = span_for_range(i1, i2)
                    diffs_out.append({
                        "type": "replace",
                        "start": start,
                        "end": end,
                        "original": orig_tok,
                        "suggestion": corr_tok,
                    })

        elif tag == "delete":
            if (i2 - i1) == 1:
                orig_tok = orig_tokens[i1]
                if is_pure_punctuation(orig_tok) or len(orig_tok) <= 2:
                    start, end = span_for_range(i1, i2)
                    diffs_out.append({
                        "type": "delete",
                        "start": start,
                        "end": end,
                        "original": orig_tok,
                        "suggestion": "",
                    })

        elif tag == "insert":
            if (j2 - j1) == 1:
                corr_tok = corr_tokens[j1]
                if is_pure_punctuation(corr_tok):
                    start, _ = span_for_range(i1, i1)
                    diffs_out.append({
                        "type": "insert",
                        "start": start,
                        "end": start,
                        "original": "",
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

    # Normal page render (GET)
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
