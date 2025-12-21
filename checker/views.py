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
# STRICT CHARWISE DIFF (DANISH-STYLE, NORWAY-SAFE)
# -------------------------------------------------

TOKEN_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]", re.UNICODE)


def find_differences_charwise(original: str, corrected: str):
    diffs = []

    orig = unicodedata.normalize("NFC", original)
    corr = unicodedata.normalize("NFC", corrected)

    # -------------------------------------------------
    # TOKENIZE
    # -------------------------------------------------

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

    # -------------------------------------------------
    # MAP TOKEN → CHAR POSITIONS (FAIL-SAFE)
    # -------------------------------------------------

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
        return []  # never guess

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------

    def span(pos, a, b):
        if a >= len(pos):
            return len(orig), len(orig)
        if a == b:
            return pos[a][0], pos[a][0]
        return pos[a][0], pos[b - 1][1]

    def core(s):
        # remove spaces + punctuation → handles "privat livet" == "privatlivet"
        return re.sub(r"[\W_]+", "", s.lower())

    def safe_word_replace(a, b):
        a0 = a.lower().strip(".,;:!?")
        b0 = b.lower().strip(".,;:!?")

        if not a0 or not b0:
            return False

        # HARD RULES (kill drift)
        if a0[0] != b0[0]:
            return False

        return difflib.SequenceMatcher(a=a0, b=b0).ratio() >= 0.9

    # -------------------------------------------------
    # DIFF
    # -------------------------------------------------

    sm = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():

        if tag == "equal":
            continue

        # -------------------------------------------------
# -------------------------------------------------
# IGNORE JOIN / SPLIT (SPACE REMOVAL / INSERTION)
        # -------------------------------------------------
        if tag == "replace" and max(i2 - i1, j2 - j1) <= 3:
            o_chunk = orig[span(o_pos, i1, i2)[0]:span(o_pos, i1, i2)[1]]
            c_chunk = corr[span(c_pos, j1, j2)[0]:span(c_pos, j1, j2)[1]]

            # If this is only a space removal/addition (compound word),
            # we ignore it completely to avoid bad suggestions.
            if core(o_chunk) == core(c_chunk):
                continue


        # -------------------------------------------------
        # SINGLE WORD REPLACE (VERY STRICT)
        # -------------------------------------------------
        if tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
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

        # -------------------------------------------------
        # INSERT / DELETE → punctuation ONLY
        # -------------------------------------------------
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
            continue

        if tag == "delete" and (i2 - i1) == 1:
            if re.fullmatch(r"[.,;:!?]+", o_tokens[i1]) or len(o_tokens[i1]) <= 2:
                s, e = span(o_pos, i1, i2)
                diffs.append({
                    "type": "delete",
                    "start": s,
                    "end": e,
                    "original": orig[s:e],
                    "suggestion": "",
                })
            continue

        # EVERYTHING ELSE → IGNORE (no red explosion)

    return diffs

def restore_original_whitespace(original: str, corrected: str) -> str:
    """
    Keeps characters from corrected, but forces whitespace
    (spaces, newlines, tabs) to match the original exactly.
    """
    out = []
    i = j = 0

    while i < len(original) and j < len(corrected):
        if original[i].isspace():
            out.append(original[i])
            i += 1
            if j < len(corrected) and corrected[j].isspace():
                j += 1
        else:
            out.append(corrected[j])
            i += 1
            j += 1

    # append remaining original whitespace if any
    while i < len(original):
        out.append(original[i])
        i += 1

    return "".join(out)

# -------------------------------------------------
# OPENAI CORRECTION (UNCHANGED)
# -------------------------------------------------

def correct_with_openai_no(text: str) -> str:
    try:
        prompt = (
            "Du er en profesjonell norsk språkvasker.\n\n"

            "ABSOLUTTE REGLER (MÅ FØLGES):\n"
            "- IKKE legg til mellomrom\n"
            "- IKKE fjern mellomrom\n"
            "- IKKE slå sammen ord\n"
            "- IKKE del ord\n"
            "- IKKE endre linjeskift eller avsnitt\n\n"

            "DU HAR LOV TIL Å RETTE:\n"
            "- stavefeil inne i samme ord\n"
            "- bøyningsfeil av samme ord\n"
            "- tegnsetting (komma, punktum osv.)\n"
            "- store og små bokstaver\n\n"

            "Behold tekstens betydning, stil og tone uendret.\n"
            "Returner KUN teksten, uten forklaring."
        )

        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )

        corrected = (r.choices[0].message.content or "").strip()
        if not corrected:
            return text

        # 🔒 FIX whitespace instead of rejecting everything
        if re.sub(r"\S", "", corrected) != re.sub(r"\S", "", text):
            logger.warning("Whitespace changed by model – repairing whitespace")
            corrected = restore_original_whitespace(text, corrected)

        return corrected

    except Exception:
        logger.exception("OpenAI error")
        return text


# -------------------------------------------------
# MAIN VIEW
# -------------------------------------------------

def index(request):
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        original = request.POST.get("text", "")

        original = unicodedata.normalize("NFC", original)

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
