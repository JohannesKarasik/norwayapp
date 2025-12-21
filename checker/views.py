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
# CHARWISE DIFF (SIMPLE, SAFE)
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

    def norm_word(tok: str) -> str:
        return tok.lower().strip(".,;:!?")

    def levenshtein(a: str, b: str) -> int:
        """Small, fast Levenshtein for short tokens."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        # Ensure a is shorter
        if len(a) > len(b):
            a, b = b, a

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i]
            for j, cb in enumerate(b, start=1):
                ins = cur[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (ca != cb)
                cur.append(min(ins, dele, sub))
            prev = cur
        return prev[-1]

    def safe_word_replace(o, c):
        """
        Allow real spelling/inflection fixes while preventing wild swaps.
        Key: short words need looser thresholds (dt->det, de->dem, job->jobb).
        """
        o0 = norm_word(o)
        c0 = norm_word(c)

        if not o0 or not c0:
            return False
        if o0 == c0:
            return False

        # Do not allow completely different starting letters (prevents many bad swaps).
        # (This also blocks og -> å, which is grammatical rather than "spelling".)
        if o0[0] != c0[0]:
            return False

        L = max(len(o0), len(c0))

        # Extra-safe allowlist patterns for common spelling expansions
        # job -> jobb, etc.
        if len(c0) == len(o0) + 1 and c0[:-1] == o0 and c0[-1] == o0[-1]:
            return True

        # Prefix expansion/contraction (det <-> dt won't match, but other common ones will)
        if (c0.startswith(o0) or o0.startswith(c0)) and abs(len(o0) - len(c0)) <= 2 and L <= 6:
            return True

        # Levenshtein-based acceptance (very good for short tokens)
        dist = levenshtein(o0, c0)
        if L <= 3:
            return dist <= 1  # dt->det (dist 1), de->dem (dist 1)
        if L <= 5:
            return dist <= 1

        # For longer words, require strong similarity
        ratio = difflib.SequenceMatcher(a=o0, b=c0).ratio()
        return ratio >= 0.88

    def is_compound_join(o_tokens, i1, c_tok):
        """
        Blocks compound joins like:
        'privat' + 'livet' -> 'privatlivet'
        even if the diff comes out as a 1->1 replace.
        """
        c0 = norm_word(c_tok)
        if not c0:
            return False

        cur = norm_word(o_tokens[i1]) if i1 < len(o_tokens) else ""
        nxt = norm_word(o_tokens[i1 + 1]) if (i1 + 1) < len(o_tokens) else ""
        prv = norm_word(o_tokens[i1 - 1]) if (i1 - 1) >= 0 else ""

        if not cur:
            return False

        if nxt and (cur + nxt) == c0:
            return True
        if prv and (prv + cur) == c0:
            return True

        if nxt and c0.startswith(cur) and c0.endswith(nxt) and len(c0) > len(cur) and len(c0) > len(nxt):
            return True
        if prv and c0.startswith(prv) and c0.endswith(cur) and len(c0) > len(prv) and len(c0) > len(cur):
            return True

        return False

    sm = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        # Ignore joins/splits entirely when they show up as multi-token changes
        if tag == "replace" and ((i2 - i1) != 1 or (j2 - j1) != 1):
            continue

        if tag == "replace":
            o_tok = o_tokens[i1]
            c_tok = c_tokens[j1]

            # Block compound joins even if they appear as 1->1 replace
            if is_compound_join(o_tokens, i1, c_tok):
                continue

            if safe_word_replace(o_tok, c_tok):
                s, e = span(o_pos, i1, i2)
                orig_span = orig[s:e]
                diffs.append({
                    "type": "replace",
                    "start": s,
                    "end": e,
                    "original": orig_span,
                    "suggestion": c_tok,
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
            continue

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
            continue

    return diffs


# -------------------------------------------------
# APPLY ONLY SAFE DIFFS (PREVENTS WORD COMBINATIONS)
# -------------------------------------------------

def apply_diffs_to_text(original: str, diffs):
    text = original

    for d in sorted(diffs, key=lambda x: (x["start"], x["end"]), reverse=True):
        s = int(d.get("start", 0))
        e = int(d.get("end", s))
        suggestion = d.get("suggestion", "")

        if s < 0 or e < 0 or s > len(text) or e > len(text) or s > e:
            continue

        text = text[:s] + suggestion + text[e:]

    return text


# -------------------------------------------------
# OPENAI CORRECTION
# -------------------------------------------------

def correct_with_openai_no(text: str) -> str:
    try:
        text = re.sub(r"\s+", " ", text).strip()

        prompt = (
            "Du er en profesjonell norsk språkvasker.\n\n"
            "VIKTIGE REGLER (MÅ FØLGES):\n"
            "- IKKE legg til nye ord\n"
            "- IKKE fjern ord\n"
            "- IKKE endre rekkefølgen på ord\n"
            "- IKKE slå sammen ord (f.eks. 'privat livet' -> 'privatlivet')\n"
            "- IKKE del ord\n\n"
            "Du kan rette:\n"
            "- stavefeil\n"
            "- grammatikk (kun ved å endre eksisterende ord)\n"
            "- bøyning (kun ved å endre eksisterende ord)\n"
            "- tegnsetting\n"
            "- store og små bokstaver\n\n"
            "Behold mening og stil.\n"
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

        raw_corrected = unicodedata.normalize("NFC", correct_with_openai_no(original))
        diffs = find_differences_charwise(original, raw_corrected)

        safe_corrected = apply_diffs_to_text(original, diffs)

        return JsonResponse({
            "original_text": original,
            "corrected_text": safe_corrected,
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
