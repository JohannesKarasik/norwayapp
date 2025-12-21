from django.shortcuts import render
from django.http import JsonResponse
from openai import OpenAI
import logging
import re
import difflib
import unicodedata

client = OpenAI()
logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]", re.UNICODE)


def find_differences_charwise(original: str, corrected: str):
    diffs = []

    orig = unicodedata.normalize("NFC", original)
    corr = unicodedata.normalize("NFC", corrected)

    def merge_punct(tokens):
        out = []
        for t in tokens:
            if out and re.match(r"^[.,;:!?]+$", t):
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

    def safe_word_replace(o, c):
        o0 = o.lower().strip(".,;:!?")
        c0 = c.lower().strip(".,;:!?")

        if not o0 or not c0:
            return False

        if o0[0] != c0[0]:
            return False

        return difflib.SequenceMatcher(a=o0, b=c0).ratio() >= 0.88

    sm = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        if tag == "replace" and ((i2 - i1) != 1 or (j2 - j1) != 1):
            continue

        if tag == "replace":
            o_tok = o_tokens[i1]
            c_tok = c_tokens[j1]

            if not c_tok:
                continue

            next_orig = o_tokens[i1 + 1] if i1 + 1 < len(o_tokens) else None
            if next_orig and c_tok == o_tok + next_orig:
                continue

            if safe_word_replace(o_tok, c_tok):
                s, e = span(o_pos, i1, i2)
                diffs.append({
                    "type": "replace",
                    "start": s,
                    "end": e,
                    "original": orig[s:e],
                    "suggestion": c_tok,
                })
            continue

        if tag == "insert" and (j2 - j1) == 1:
            tok = c_tokens[j1]
            if tok and re.match(r"^[.,;:!?]+$", tok):
                s, _ = span(o_pos, i1, i1)
                diffs.append({
                    "type": "insert",
                    "start": s,
                    "end": s,
                    "original": "",
                    "suggestion": tok,
                })
            continue

        if tag == "delete" and (i2 - i1) == 1:
            tok = o_tokens[i1]
            if tok and re.match(r"^[.,;:!?]+$", tok):
                s, e = span(o_pos, i1, i2)
                diffs.append({
                    "type": "delete",
                    "start": s,
                    "end": e,
                    "original": orig[s:e],
                    "suggestion": "",
                })
            continue

    return diffs


def correct_with_openai_no(text: str) -> str:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du er en profesjonell norsk språkvasker.\n\n"
                        "Du kan rette:\n"
                        "- stavefeil\n"
                        "- grammatikk\n"
                        "- bøyning\n"
                        "- tegnsetting\n"
                        "- store og små bokstaver\n\n"
                        "Returner KUN teksten."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
        )

        return (r.choices[0].message.content or "").strip() or text

    except Exception:
        logger.exception("OpenAI error")
        return text


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




from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import redirect


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
