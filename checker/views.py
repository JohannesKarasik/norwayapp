# checker/views.py
from django.shortcuts import render
from django.http import JsonResponse
from openai import OpenAI
import logging

client = OpenAI()
logger = logging.getLogger(__name__)


def correct_with_openai_sv(text: str) -> str:
    try:
        system_prompt = (
            "Du är en professionell svensk språkre­daktör. "
            "Din uppgift är att korrigera ALLA fel i stavning, grammatik, "
            "ordföljd och skiljetecken (särskilt kommatecken). "
            "Behåll betydelsen exakt. "
            "Returnera ENDAST den korrigerade texten."
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


def index(request):
    # Handle AJAX correction (allow anonymous users)
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        text = (request.POST.get("text") or "").strip()

        if not text:
            return JsonResponse({
                "original_text": "",
                "corrected_text": "",
            })

        corrected = correct_with_openai_sv(text)

        return JsonResponse({
            "original_text": text,
            "corrected_text": corrected,
        })

    # Normal page render (GET)
    return render(request, "checker/index.html")



from django.contrib.auth.models import User
from django.contrib.auth import login
from django.shortcuts import redirect
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
from django.shortcuts import redirect
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
        username=email,   # username IS email in your system
        password=password
    )

    if user is None:
        messages.error(request, "Fel e-post eller lösenord.")
        return redirect(request.POST.get("next", "/"))

    login(request, user)
    return redirect(request.POST.get("next", "/"))



from django.contrib.auth import logout
from django.shortcuts import redirect

def logout_view(request):
    if request.method == "POST":
        logout(request)
    return redirect("/")
