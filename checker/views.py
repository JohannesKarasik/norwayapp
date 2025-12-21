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
# HELPERS (TOP-LEVEL — IMPORTANT)
# -------------------------------------------------

def norm_word(tok: str) -> str:
    return tok.lower().strip(".,;:!?")

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]

def safe_word_replace(o, c):
    """
    General spelling / inflection safety:
    - Allows: tekst→tekster, de→dem, jobb→job
    - Blocks large rewrites
    - Language-agnostic
    """
    o0 = norm_word(o)
    c0 = norm_word(c)

    if not o0 or not c0 or o0 == c0:
        return False

    L = max(len(o0), len(c0))

    # Block extreme jumps
    if abs(len(o0) - len(c0)) > 3 and L <= 8:
        return False
    if abs(len(o0) - len(c0)) > 4:
        return False

    # Anchor for non-trivial words
    if L >= 4:
        if not (o0[:2] == c0[:2] or o0[-2:] == c0[-2:]):
            return False

    dist = levenshtein(o0, c0)

    if L <= 3:
        return dist <= 1
    if L <= 6:
        return dist <= 2
    if L <= 10:
        return dist <= 3

    ratio = difflib.SequenceMatcher(a=o0, b=c0).ratio()
    return ratio >= 0.82

def is_compound_join(o_tokens, i1, c_tok):
    """
    Blocks: privat + livet → privatlivet
    """
    c0 = norm_word(c_tok)
    if not c0:
        return False

    cur = norm_word(o_tokens[i1]) if i1 < len(o_tokens) else ""
    nxt = norm_word(o_tokens[i1 + 1]) if i1 + 1 < len(o_tokens) else ""
    prv = norm_word(o_tokens[i1 - 1]) if i1 - 1 >= 0 else ""

    return (
        (nxt and c0 == cur + nxt) or
        (prv and c0 == prv + cur)
    )

# -------------------------------------------------
# DIFF ENGINE
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

    sm = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        # Ignore joins/splits
        if tag == "replace" and ((i2 - i1) != 1 or (j2 - j1) != 1):
            continue

        if tag == "replace":
            o_tok = o_tokens[i1]
            c_tok = c_tokens[j1]

            if is_compound_join(o_tokens, i1, c_tok):
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

        elif tag == "insert" and (j2 - j1) == 1:
            if re.fullmatch(r"[.,;:!?]+", c_tokens[j1]):
                s, _ = span(o_pos, i1, i1)
                diffs.append({
                    "type": "insert",
                    "start": s,
                    "end": s,
                    "original": "",
                    "suggestion": c_tokens[j1],
                })

        elif tag == "delete" and (i2 - i1) == 1:
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
# APPLY SAFE DIFFS
# -------------------------------------------------

def apply_diffs_to_text(original: str, diffs):
    text = original
    for d in sorted(diffs, key=lambda x: (x["start"], x["end"]), reverse=True):
        text = text[:d["start"]] + d["suggestion"] + text[d["end"]:]
    return text

# -------------------------------------------------
# OPENAI
# -------------------------------------------------

def correct_with_openai_no(text: str) -> str:
    try:
        text = re.sub(r"\s+", " ", text).strip()

        prompt = (
            "Du er en profesjonell norsk språkvasker.\n\n"
            "REGLER:\n"
            "- IKKE legg til eller fjern ord\n"
            "- IKKE endre rekkefølge\n"
            "- IKKE slå sammen eller dele ord\n\n"
            "Du kan rette:\n"
            "- stavefeil\n"
            "- bøyning\n"
            "- tegnsetting\n"
            "- store/små bokstaver\n\n"
            "Returner KUN teksten."
        )

        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )

        return (r.choices[0].message.content or "").strip() or text
    except Exception:
        logger.exception("OpenAI error")
        return text

# -------------------------------------------------
# VIEW
# -------------------------------------------------

def index(request):
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        original = unicodedata.normalize("NFC", request.POST.get("text", ""))

        corrected_raw = correct_with_openai_no(original)
        diffs = find_differences_charwise(original, corrected_raw)
        safe_corrected = apply_diffs_to_text(original, diffs)

        return JsonResponse({
            "original_text": original,
            "corrected_text": safe_corrected,
            "differences": diffs,
            "error_count": len(diffs),
        })

    return render(request, "checker/index.html")

# -------------------------------------------------
# AUTH
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
