from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import logging
import re
import difflib
import unicodedata

from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages

client = OpenAI()
logger = logging.getLogger(__name__)

# =================================================
# TOKENIZATION (EXACT DANISH STRATEGY + WHITESPACE)
# =================================================
# KEY INSIGHT:
# Danish works because whitespace is tokenized.
# That forces "privat livet" -> "privatlivet" to become a MULTI-TOKEN replace,
# which we can safely ignore.
TOKEN_RE = re.compile(
    r"[\s\u200B\uFEFF]+|\w+(?:-\w+)*|[^\w\s]",
    re.UNICODE,
)

# =================================================
# APPLY ONLY SAFE DIFFS (MODEL IS NEVER TRUSTED)
# =================================================
def apply_diffs(original: str, diffs: list[dict]) -> str:
    out = original
    for d in sorted(diffs, key=lambda x: (x["start"], x["end"]), reverse=True):
        logger.debug("APPLY DIFF: %s", d)
        if d["type"] == "replace":
            out = out[:d["start"]] + d["suggestion"] + out[d["end"]:]
        elif d["type"] == "delete":
            out = out[:d["start"]] + out[d["end"]:]
        elif d["type"] == "insert":
            out = out[:d["start"]] + d["suggestion"] + out[d["start"]:]
    return out

# =================================================
# CHARWISE DIFF (DANISH LOGIC + HARD GUARDS)
# =================================================
def find_differences_charwise(original: str, corrected: str):
    diffs = []

    orig = unicodedata.normalize("NFC", original)
    corr = unicodedata.normalize("NFC", corrected)

    logger.debug("ORIGINAL TEXT: %r", orig)
    logger.debug("MODEL TEXT:    %r", corr)

    def is_ws(tok: str) -> bool:
        return bool(re.fullmatch(r"[\s\u200B\uFEFF]+", tok))

    def merge_punct(tokens):
        out = []
        for t in tokens:
            if out and not is_ws(out[-1]) and re.fullmatch(r"[,.:;!?]", t):
                out[-1] += t
            else:
                out.append(t)
        return out

    o_tokens = merge_punct(TOKEN_RE.findall(orig))
    c_tokens = merge_punct(TOKEN_RE.findall(corr))

    logger.debug("ORIGINAL TOKENS: %s", o_tokens)
    logger.debug("MODEL TOKENS:    %s", c_tokens)

    # Map tokens to character spans
    o_pos = []
    cursor = 0
    for t in o_tokens:
        idx = orig.find(t, cursor)
        if idx == -1:
            logger.error("FAILED TO MAP TOKEN: %r", t)
            return []
        o_pos.append((idx, idx + len(t)))
        cursor = idx + len(t)

    def span(i1, i2):
        if i1 >= len(o_pos):
            return len(orig), len(orig)
        if i1 == i2:
            return o_pos[i1][0], o_pos[i1][0]
        return o_pos[i1][0], o_pos[i2 - 1][1]

    def small_edit(a, b):
        ratio = difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()
        logger.debug("SIMILARITY %r -> %r = %.3f", a, b, ratio)
        return ratio >= 0.88

    def longer_than_original(a, b):
        return len(b.strip()) > len(a.strip())

    sm = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        logger.debug(
            "OPCODE %-7s O[%d:%d]=%r C[%d:%d]=%r",
            tag,
            i1,
            i2,
            o_tokens[i1:i2],
            j1,
            j2,
            c_tokens[j1:j2],
        )

        if tag == "equal":
            continue

        # 🔥 THIS IS THE CORE FIX
        # Any join/split MUST be multi-token.
        # Those are NEVER surfaced.
        if tag == "replace" and ((i2 - i1) != 1 or (j2 - j1) != 1):
            logger.debug("SKIP: multi-token replace (join/split)")
            continue

        if tag == "replace":
            o = o_tokens[i1]
            c = c_tokens[j1]

            if is_ws(o) or is_ws(c):
                logger.debug("SKIP: whitespace token")
                continue

            if longer_than_original(o, c):
                logger.debug("SKIP: longer suggestion %r -> %r", o, c)
                continue

            if not small_edit(o, c):
                logger.debug("SKIP: edit too large")
                continue

            s, e = span(i1, i2)
            diffs.append({
                "type": "replace",
                "start": s,
                "end": e,
                "original": orig[s:e],
                "suggestion": c,
            })
            logger.debug("ACCEPT REPLACE: %r -> %r", o, c)

        elif tag == "insert":
            if (j2 - j1) == 1 and re.fullmatch(r"[,.:;!?]", c_tokens[j1]):
                s, _ = span(i1, i1)
                diffs.append({
                    "type": "insert",
                    "start": s,
                    "end": s,
                    "original": "",
                    "suggestion": c_tokens[j1],
                })
                logger.debug("ACCEPT INSERT: %r", c_tokens[j1])

        elif tag == "delete":
            if (i2 - i1) == 1 and re.fullmatch(r"[,.:;!?]", o_tokens[i1]):
                s, e = span(i1, i2)
                diffs.append({
                    "type": "delete",
                    "start": s,
                    "end": e,
                    "original": orig[s:e],
                    "suggestion": "",
                })
                logger.debug("ACCEPT DELETE: %r", o_tokens[i1])

    logger.debug("FINAL DIFFS: %s", diffs)
    return diffs

# =================================================
# OPENAI CORRECTION (MODEL IS UNTRUSTED)
# =================================================
def correct_with_openai_no(text: str) -> str:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du er en norsk språkvasker.\n"
                        "IKKE slå sammen eller del ord.\n"
                        "IKKE endre mellomrom.\n"
                        "Kun små rettelser i samme ord."
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

# =================================================
# MAIN VIEW
# =================================================
def index(request):
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        original = request.POST.get("text", "")

        model_text = correct_with_openai_no(original)
        diffs = find_differences_charwise(original, model_text)
        safe_text = apply_diffs(original, diffs)

        return JsonResponse({
            "original_text": original,
            "corrected_text": safe_text,
            "differences": diffs,
            "error_count": len(diffs),
        })

    return render(request, "checker/index.html")

# =================================================
# AUTH (UNCHANGED)
# =================================================
def register(request):
    if request.method != "POST":
        return redirect("index")

    user = User.objects.create_user(
        username=request.POST.get("email"),
        email=request.POST.get("email"),
        password=request.POST.get("password"),
        first_name=request.POST.get("name") or "",
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
    if not user:
        messages.error(request, "Feil e-post eller passord.")
        return redirect("/")
    login(request, user)
    return redirect("/")

def logout_view(request):
    logout(request)
    return redirect("/")
