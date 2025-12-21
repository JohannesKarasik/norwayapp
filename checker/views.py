from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import logging
import re
import difflib
import unicodedata

client = OpenAI()
logger = logging.getLogger(__name__)



def correct_with_openai_no(text: str) -> str:
    """
    Returns a corrected version of the text where:
    - NO words are added or removed
    - Word order is identical
    - Only spelling and punctuation attached to a word may change
    """
    try:
        # ✅ COLLAPSE WHITESPACE (self-defensive)
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
            "- legge til eller fjerne tegnsetting SOM ER EN DEL AV ORDET "
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

        # 🔐 HARD SAFETY: word count must match
        if len(corrected.split()) != len(text.split()):
            logger.warning("Word count mismatch – falling back to original text")
            return text

        return corrected if corrected else text

    except Exception:
        logger.exception("OpenAI error")
        return text


def find_differences_charwise(original: str, corrected: str):
    diffs_out = []

    orig = unicodedata.normalize("NFC", original)
    corr = unicodedata.normalize("NFC", corrected)

    matcher = difflib.SequenceMatcher(a=orig, b=corr)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue

        orig_chunk = orig[i1:i2]
        corr_chunk = corr[j1:j2]

        # 🔒 Skip large or semantic rewrites
        if len(orig_chunk.split()) != 1 or len(corr_chunk.split()) != 1:
            continue

        a_core = orig_chunk.lower().strip(".,;:!?")
        b_core = corr_chunk.lower().strip(".,;:!?")

        if a_core == b_core:
            diffs_out.append({
                "type": "replace",
                "start": i1,
                "end": i2,
                "original": orig_chunk,
                "suggestion": corr_chunk,
            })
            continue

        ratio = difflib.SequenceMatcher(a=a_core, b=b_core).ratio()
        if ratio >= 0.8:
            diffs_out.append({
                "type": "replace",
                "start": i1,
                "end": i2,
                "original": orig_chunk,
                "suggestion": corr_chunk,
            })

    return diffs_out

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

        # ✅ COLLAPSE ONCE, EARLY, AND USE EVERYWHERE
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
