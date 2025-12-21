from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import logging
import re
import difflib
import unicodedata

client = OpenAI()
logger = logging.getLogger(__name__)

# -------------------------------------------------
# TOKENIZATION (DANISH-PROVEN)
# -------------------------------------------------

TOKEN_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]", re.UNICODE)


# -------------------------------------------------
# CHARWISE DIFF (UNCHANGED, SAFE)
# -------------------------------------------------

def find_differences_charwise(original: str, corrected: str):
    diffs = []

    orig = unicodedata.normalize("NFC", original)
    corr = unicodedata.normalize("NFC", corrected)

    def merge_punct(tokens):
        out = []
        for t in tokens:
            if out and re.fullmatch(r"[.,;:!?]", t):
                out[-1] += t
            else:
                out.append(t)
        return out

    o_tokens = merge_punct(TOKEN_RE.findall(orig))
    c_tokens = merge_punct(TOKEN_RE.findall(corr))

    def map_positions(text, tokens):
        pos = []
        i = 0
        for t in tokens:
            p = text.find(t, i)
            if p == -1:
                return None
            pos.append((p, p + len(t)))
            i = p + len(t)
        return pos

    o_pos = map_positions(orig, o_tokens)
    c_pos = map_positions(corr, c_tokens)

    if o_pos is None or c_pos is None:
        return []

    def span(pos, a, b):
        if a >= len(pos):
            return len(orig), len(orig)
        if a == b:
            return pos[a][0], pos[a][0]
        return pos[a][0], pos[b - 1][1]

    def safe_word_replace(a, b):
        a0 = a.lower().strip(".,;:!?")
        b0 = b.lower().strip(".,;:!?")

        if not a0 or not b0:
            return False
        if a0[0] != b0[0]:
            return False

        return difflib.SequenceMatcher(a=a0, b=b0).ratio() >= 0.88

    sm = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        # Ignore join/split completely
        if tag == "replace" and ((i2 - i1) != 1 or (j2 - j1) != 1):
            continue

        if tag == "replace":
            o = o_tokens[i1]
            c = c_tokens[j1]

            if safe_word_replace(o, c):
                s, e = span(o_pos, i1, i2)
                diffs.append({
                    "type": "replace",
                    "start": s,
                    "end": e,
                    "original": orig[s:e],
                    "suggestion": c,
                })
            continue

        if tag == "insert" and (j2 - j1) == 1:
            if re.fullmatch(r"[.,;:!?]+", c_tokens[j1]):
                s, _ = span(o_pos, i1, i1)
                diffs.append({
                    "type": "insert",
                    "start": s,
                    "end": s,
                    "original": "",
                    "suggestion": c_tokens[j1],
                })

        if tag == "delete" and (i2 - i1) == 1:
            if re.fullmatch(r"[.,;:!?]+", o_tokens[i1]):
                s, e = span(o_pos, i1, i2)
                diffs.append({
                    "type": "delete",
                    "start": s,
                    "end": e,
                    "original": orig[s:e],
                    "suggestion": "",
                })

    return diffs


# -------------------------------------------------
# OPENAI CORRECTION (DANISH TWO-PASS SYSTEM)
# -------------------------------------------------
def correct_with_openai_no(text: str) -> str:
    try:
        base_prompt = (
            "Du er en profesjonell norsk språkvasker.\n"
            "Returner teksten i korrekt norsk bokmål.\n\n"
            "Du kan rette:\n"
            "- stavefeil\n"
            "- grammatikk\n"
            "- bøyning\n"
            "- tegnsetting\n"
            "- store og små bokstaver\n\n"
            "Behold betydning og stil.\n"
            "Returner KUN teksten."
        )

        strict_prompt = (
            "Du er en ekstremt streng norsk språkkorrektør.\n\n"
            "ABSOLUTTE REGLER:\n"
            "- IKKE legg til mellomrom\n"
            "- IKKE fjern mellomrom\n"
            "- IKKE slå sammen ord\n"
            "- IKKE del ord\n"
            "- IKKE endre rekkefølge på ord\n\n"
            "Du har KUN lov til å rette:\n"
            "- stavefeil inne i samme ord\n"
            "- små bøyningsfeil\n"
            "- tegnsetting\n"
            "- store/små bokstaver\n\n"
            "Returner KUN teksten."
        )

        # ---- PASS 1: full correction
        r1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )

        corrected = (r1.choices[0].message.content or "").strip()
        if not corrected:
            return text

        # ---- WHITESPACE CHECK
        if re.sub(r"\S", "", corrected) != re.sub(r"\S", "", text):
            # ---- PASS 2: STRICT MODE
            r2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": strict_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )

            strict_corrected = (r2.choices[0].message.content or "").strip()

            # 🔒 FINAL HARD RULE
            if not strict_corrected:
                return text

            if re.sub(r"\S", "", strict_corrected) != re.sub(r"\S", "", text):
                logger.warning("Whitespace change detected – skipping corrections")
                return text

            return strict_corrected

        return corrected

    except Exception:
        logger.exception("OpenAI error")
        return text


# -------------------------------------------------
# MAIN VIEW
# -------------------------------------------------

def index(request):
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        original = unicodedata.normalize("NFC", request.POST.get("text", ""))

        if not original.strip():
            return JsonResponse({
                "original_text": "",
                "corrected_text": "",
                "differences": [],
                "error_count": 0,
            })

        corrected = unicodedata.normalize("NFC", correct_with_openai_no(original))
        diffs = find_differences_charwise(original, corrected)

        return JsonResponse({
            "original_text": original,
            "corrected_text": corrected,
            "differences": diffs,
            "error_count": len(diffs),
        })

    return render(request, "checker/index.html")


# -------------------------------------------------
# AUTH (UNCHANGED)
# -------------------------------------------------

from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages


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
