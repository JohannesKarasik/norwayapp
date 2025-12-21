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
# STRICT CHARWISE DIFF (DANISH-PROVEN LOGIC)
# - Only highlight SAFE 1-to-1 token replacements
# - Ignore any join/split (space add/remove) by ignoring non 1-to-1 replaces
# - No whitespace hacks / no post-repair
# -------------------------------------------------

TOKEN_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]", re.UNICODE)


def find_differences_charwise(original: str, corrected: str):
    diffs = []

    orig = unicodedata.normalize("NFC", original)
    corr = unicodedata.normalize("NFC", corrected)

    def merge_punct(tokens):
        out = []
        for t in tokens:
            # attach punctuation to previous token (same as Danish logic)
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
        return []  # fail-safe: never guess

    def span(pos, a, b, text_len):
        if a >= len(pos):
            return text_len, text_len
        if a == b:
            return pos[a][0], pos[a][0]
        return pos[a][0], pos[b - 1][1]

    def safe_word_replace(a, b):
        a0 = a.lower().strip(".,;:!?")
        b0 = b.lower().strip(".,;:!?")

        if not a0 or not b0:
            return False

        # HARD RULES (kill drift)
        if a0[0] != b0[0]:
            return False

        return difflib.SequenceMatcher(a=a0, b=b0).ratio() >= 0.88

    sm = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        # -------------------------------------------------
        # IGNORE JOIN / SPLIT (space changes) AND any multi-token replace
        # -------------------------------------------------
        if tag == "replace" and ((i2 - i1) != 1 or (j2 - j1) != 1):
            continue

        # -------------------------------------------------
        # SINGLE WORD REPLACE (safe only)
        # -------------------------------------------------
        if tag == "replace":
            o = o_tokens[i1]
            c = c_tokens[j1]

            if safe_word_replace(o, c):
                s, e = span(o_pos, i1, i2, len(orig))
                diffs.append({
                    "type": "replace",
                    "start": s,
                    "end": e,
                    "original": orig[s:e],
                    "suggestion": c,
                })
            continue

        # -------------------------------------------------
        # INSERT punctuation only (rare due to merge_punct, but keep)
        # -------------------------------------------------
        if tag == "insert" and (j2 - j1) == 1:
            if re.fullmatch(r"[.,;:!?]+", c_tokens[j1]):
                s, _ = span(o_pos, i1, i1, len(orig))
                diffs.append({
                    "type": "insert",
                    "start": s,
                    "end": s,
                    "original": "",
                    "suggestion": c_tokens[j1],
                })
            continue

        # -------------------------------------------------
        # DELETE punctuation only (and tiny tokens) (Danish-safe behavior)
        # -------------------------------------------------
        if tag == "delete" and (i2 - i1) == 1:
            tok = o_tokens[i1]
            if re.fullmatch(r"[.,;:!?]+", tok) or len(tok) <= 2:
                s, e = span(o_pos, i1, i2, len(orig))
                diffs.append({
                    "type": "delete",
                    "start": s,
                    "end": e,
                    "original": orig[s:e],
                    "suggestion": "",
                })
            continue

        # EVERYTHING ELSE → IGNORE (prevents red explosions)

    return diffs


# -------------------------------------------------
# OPENAI CORRECTION (DANISH-STYLE: no whitespace hacks)
# -------------------------------------------------

def correct_with_openai_no(text: str) -> str:
    """
    Full Norwegian correction (Bokmål) like Danish:
    - spelling, grammar, inflection, punctuation, capitalization, commas
    - keep meaning/tone; avoid stylistic rewriting
    - DOES NOT do any post whitespace enforcement (same as Danish)
    """
    try:
        base_prompt = (
            "Du er en profesjonell norsk språkvasker (bokmål).\n\n"
            "Oppgave:\n"
            "- Returner teksten i perfekt norsk bokmål.\n"
            "- Rett stavefeil, grammatikk, bøyning, tegnsetting, store/små bokstaver og kommatering "
            "(inkludert komma før leddsetninger / startkomma der det passer).\n\n"
            "Regler:\n"
            "- Behold betydning og tone.\n"
            "- Ikke legg til nye setninger.\n"
            "- Ikke fjern innhold.\n"
            "- Unngå stilistisk omskriving.\n\n"
            "Returner KUN den korrigerte teksten, uten forklaring."
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

        # Danish-style retry if no visible change
        if corrected and corrected.strip() == text.strip():
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": base_prompt
                        + "\n\nVær ekstra nøye: finn også små komma- og tegnsettingsfeil.",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            corrected2 = (resp2.choices[0].message.content or "").strip()
            if corrected2:
                corrected = corrected2

        return corrected or text

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
