from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import logging
import re
import difflib

client = OpenAI()
logger = logging.getLogger(__name__)




def correct_with_openai_no(text: str) -> str:
    """
    Correct Norwegian text WITHOUT changing:
    - word count
    - word order
    - whitespace
    """
    try:
        # ✅ Canonical whitespace
        text = re.sub(r"\s+", " ", text).strip()

        system_prompt = (
            "Du er en profesjonell norsk språkkorrektør.\n\n"
            "VIKTIGE REGLER (MÅ FØLGES):\n"
            "- IKKE legg til nye ord\n"
            "- IKKE fjern ord\n"
            "- IKKE endre rekkefølgen på ord\n"
            "- IKKE del eller slå sammen ord\n"
            "- IKKE endre mellomrom eller linjeskift\n\n"
            "Du har KUN lov til å:\n"
            "- rette stavefeil INNE I eksisterende ord\n"
            "- legge til eller fjerne tegnsetting SOM ER EN DEL AV ORDET\n\n"
            "Hvis en feil krever omskriving, LA DEN STÅ URØRT.\n\n"
            "Returner KUN teksten."
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

        # HARD SAFETY: word count must match
        if len(corrected.split()) != len(text.split()):
            logger.warning("Word count mismatch – disabling correction")
            return text

        return corrected if corrected else text

    except Exception:
        logger.exception("OpenAI error")
        return text

def find_differences_charwise(original: str, corrected: str):
    """
    Bullet-proof diff:
    - Compares word-by-word ONLY
    - Aborts immediately if alignment breaks
    """

    diffs = []

    orig_words = original.split(" ")
    corr_words = corrected.split(" ")

    # ❌ Abort if alignment is impossible
    if len(orig_words) != len(corr_words):
        return []

    cursor = 0

    for orig, corr in zip(orig_words, corr_words):
        start = cursor
        end = start + len(orig)

        if orig == corr:
            cursor = end + 1
            continue

        # Strip punctuation for comparison
        o_core = orig.lower().strip(".,;:!?")
        c_core = corr.lower().strip(".,;:!?")

        # Allow only small spelling fixes
        if difflib.SequenceMatcher(a=o_core, b=c_core).ratio() < 0.8:
            # ❌ Abort everything — alignment is unsafe
            return []

        diffs.append({
            "type": "replace",
            "start": start,
            "end": end,
            "original": orig,
            "suggestion": corr,
        })

        cursor = end + 1

    return diffs

def index(request):
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        raw_text = (request.POST.get("text") or "").strip()

        if not raw_text:
            return JsonResponse({
                "original_text": "",
                "corrected_text": "",
                "differences": [],
                "error_count": 0,
            })

        # ✅ Canonical text
        collapsed_text = re.sub(r"\s+", " ", raw_text).strip()

        corrected = correct_with_openai_no(collapsed_text)
        differences = find_differences_charwise(collapsed_text, corrected)

        return JsonResponse({
            "original_text": collapsed_text,
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
