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

# -------------------------------------------------
# TOKENIZATION (DANISH-PROVEN) + WHITESPACE TOKENS
# -------------------------------------------------
# IMPORTANT: include whitespace tokens so joins/splits ("privat livet" -> "privatlivet")
# become MULTI-TOKEN replace ops and are safely ignored.
TOKEN_RE = re.compile(r"[\s\u200B\uFEFF]+|\w+(?:-\w+)*|[^\w\s]", re.UNICODE)


# -------------------------------------------------
# APPLY ONLY SAFE DIFFS BACK ONTO ORIGINAL
# (so corrected_text can NEVER merge/split words)
# -------------------------------------------------
def apply_diffs(original: str, diffs: list[dict]) -> str:
    out = original

    # apply from end -> start so indices remain valid
    diffs_sorted = sorted(diffs, key=lambda d: (d["start"], d["end"]), reverse=True)

    for d in diffs_sorted:
        t = d["type"]
        s = int(d["start"])
        e = int(d["end"])

        if t == "replace":
            out = out[:s] + d["suggestion"] + out[e:]
        elif t == "delete":
            out = out[:s] + out[e:]
        elif t == "insert":
            out = out[:s] + d["suggestion"] + out[s:]

    return out


# -------------------------------------------------
# CHARWISE DIFF (DANISH STYLE, WHITESPACE-SAFE)
# + SIMPLE RULE: never accept a suggestion that is "longer"
# -------------------------------------------------
def find_differences_charwise(original: str, corrected: str):
    diffs_out = []

    orig_text = unicodedata.normalize("NFC", original)
    corr_text = unicodedata.normalize("NFC", corrected)

    def is_ws(tok: str) -> bool:
        return bool(re.fullmatch(r"[\s\u200B\uFEFF]+", tok))

    def merge_punctuation(tokens: list[str]) -> list[str]:
        merged = []
        for tok in tokens:
            # attach punctuation to previous NON-whitespace token only
            if merged and (not is_ws(merged[-1])) and re.fullmatch(r"[,.:;!?]", tok):
                merged[-1] += tok
            else:
                merged.append(tok)
        return merged

    orig_tokens = merge_punctuation(TOKEN_RE.findall(orig_text))
    corr_tokens = merge_punctuation(TOKEN_RE.findall(corr_text))

    # map each original token back to char positions
    orig_positions = []
    cursor = 0
    for tok in orig_tokens:
        start_idx = orig_text.find(tok, cursor)
        if start_idx == -1:
            return []  # fail-safe: never guess
        end_idx = start_idx + len(tok)
        orig_positions.append((start_idx, end_idx))
        cursor = end_idx

    def span_for_range(i_start, i_end_exclusive):
        if i_start >= len(orig_positions):
            return len(orig_text), len(orig_text)
        if i_start == i_end_exclusive:
            start_i, _ = orig_positions[i_start]
            return start_i, start_i
        start_char = orig_positions[i_start][0]
        end_char = orig_positions[i_end_exclusive - 1][1]
        return start_char, end_char

    def is_pure_punctuation(tok: str) -> bool:
        return bool(re.fullmatch(r"[,.:;!?]+", tok))

    # danish helper (but with stricter threshold to avoid drift)
    def tokens_are_small_edit(a: str, b: str) -> bool:
        a_low = a.lower()
        b_low = b.lower()

        # accept if only case/punctuation changed
        if a_low.strip(",.;:!?") == b_low.strip(",.;:!?"):
            return True

        ratio = difflib.SequenceMatcher(a=a_low, b=b_low).ratio()
        return ratio >= 0.88

    # SIMPLE HARD RULE YOU ASKED FOR:
    # if suggestion is longer than original token -> discard (prevents "privat" -> "privatlivet")
    def longer_than_original(orig_tok: str, sugg_tok: str) -> bool:
        o = orig_tok.strip()
        s = sugg_tok.strip()
        return len(s) > len(o)

    sm = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        if tag == "replace":
            # ignore whitespace-only changes completely
            if all(is_ws(t) for t in orig_tokens[i1:i2]) and all(is_ws(t) for t in corr_tokens[j1:j2]):
                continue

            # only allow 1-to-1 token edits
            if (i2 - i1) == 1 and (j2 - j1) == 1:
                orig_tok = orig_tokens[i1]
                corr_tok = corr_tokens[j1]

                # never surface whitespace edits
                if is_ws(orig_tok) or is_ws(corr_tok):
                    continue

                if tokens_are_small_edit(orig_tok, corr_tok):
                    # hard length guard (your rule)
                    if longer_than_original(orig_tok, corr_tok):
                        continue

                    start_char, end_char = span_for_range(i1, i2)
                    diffs_out.append({
                        "type": "replace",
                        "start": start_char,
                        "end": end_char,
                        "original": orig_text[start_char:end_char],
                        "suggestion": corr_tok,
                    })
            # else: multi-token rewrites (includes joins/splits like "privat livet"->"privatlivet") -> ignore
            continue

        elif tag == "delete":
            # only surface delete if it's 1 token and punctuation-ish
            if (i2 - i1) == 1:
                orig_tok = orig_tokens[i1]
                if is_ws(orig_tok):
                    continue
                if is_pure_punctuation(orig_tok) or len(orig_tok) <= 2:
                    start_char, end_char = span_for_range(i1, i2)
                    diffs_out.append({
                        "type": "delete",
                        "start": start_char,
                        "end": end_char,
                        "original": orig_text[start_char:end_char],
                        "suggestion": "",
                    })

        elif tag == "insert":
            # only surface insert if it's 1 token and punctuation
            if (j2 - j1) == 1:
                corr_tok = corr_tokens[j1]
                if is_ws(corr_tok):
                    continue
                if is_pure_punctuation(corr_tok):
                    start_char, _ = span_for_range(i1, i1)
                    diffs_out.append({
                        "type": "insert",
                        "start": start_char,
                        "end": start_char,
                        "original": "",
                        "suggestion": corr_tok,
                    })

    return diffs_out


# -------------------------------------------------
# OPENAI CORRECTION
# (we do NOT trust it for whitespace safety; diffs decide what gets applied)
# -------------------------------------------------
def correct_with_openai_no(text: str) -> str:
    try:
        prompt = (
            "Du er en profesjonell norsk språkvasker.\n\n"
            "VIKTIG:\n"
            "- Ikke slå sammen ord.\n"
            "- Ikke del ord.\n"
            "- Ikke endre rekkefølgen på ord.\n"
            "- Ikke endre mellomrom eller linjeskift.\n\n"
            "Du kan rette:\n"
            "- stavefeil inne i samme ord\n"
            "- bøyningsfeil\n"
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

        # model suggestion (may include merges/splits)
        model_corrected = unicodedata.normalize("NFC", correct_with_openai_no(original))

        # compute SAFE diffs (ignores whitespace changes and ignores overlong suggestions)
        diffs = find_differences_charwise(original, model_corrected)

        # build corrected_text by applying ONLY safe diffs onto the original
        safe_corrected = unicodedata.normalize("NFC", apply_diffs(original, diffs))

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
def register(request):
    if request.method != "POST":
        return redirect("index")

    email = request.POST.get("email")
    password = request.POST.get("password")
    name = request.POST.get("name")

    if not email or not password:
        messages.error(request, "Fyll inn e-post og passord.")
        return redirect("/")

    if User.objects.filter(username=email).exists():
        messages.error(request, "E-post finnes allerede.")
        return redirect("/")

    user = User.objects.create_user(
        username=email,
        email=email,
        password=password,
        first_name=name or "",
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
