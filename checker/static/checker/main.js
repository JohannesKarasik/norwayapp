(function () {
  console.log("✅ main.js loaded");
  document.addEventListener("click", (e) => {
    console.log("🌍 GLOBAL CLICK:", e.target);
  });

  function openAuthModal(){
    window.openAuthModal && window.openAuthModal();
  }
  
  
  const MAX_WORDS = 800;

  function countWords(text) {
    return (text.trim().match(/\S+/g) || []).length;
  }
  


  const errorEl = document.getElementById("length-error");

  function enforceLengthLimit(text) {
    const wordCount = countWords(text);
    const tooLong = wordCount > MAX_WORDS;
  
    submit.disabled = tooLong;
    submit.classList.toggle("is-disabled", tooLong);
  
    editor.classList.toggle("is-too-long", tooLong);
  
    if (errorEl) {
      errorEl.hidden = !tooLong;
      errorEl.textContent =
        "Texten är för lång. Du kan rätta max 800 ord åt gången.";
    }
  }
  
  
  

  const $ = (s, scope = document) => scope.querySelector(s);

  const editor   = $("#text-editor");
  const wordsEl  = $("#words");
  const charsEl  = $("#chars");
  const form     = $("#form");
  const submit   = $("#submit");
  const clearBtn = $("#clear");

  if (!editor || !form) return;

  let lastPlainText = "";
  let activeError = null;

  /* -------------------------------
     TEXT HELPERS
  --------------------------------*/
  function escapeHTML(str) {
    return (str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function htmlWithNewlines(str) {
    return escapeHTML(str).replace(/\n/g, "<br>");
  }

  function getPlainText() {
    return (editor.innerText || "").replace(/\u00A0/g, " ");
  }

  function updateCounts(text) {
    const t = text || "";
    wordsEl.textContent = (t.trim().match(/\S+/g) || []).length;
    charsEl.textContent = t.length;
  }

  /* -------------------------------
     WORD-LEVEL DIFF
  --------------------------------*/
  function tokenizeWords(text) {
    const tokens = [];
    // ✅ include optional punctuation as part of the "word" so apply actually changes text
    const regex = /\p{L}+[.,;:!?]?/gu;
    let match;

    while ((match = regex.exec(text))) {
      tokens.push({
        word: match[0],
        start: match.index,
        end: match.index + match[0].length,
      });
    }
    return tokens;
  }

  function getWordDiffs(original, corrected) {
    const oWords = tokenizeWords(original);
    const cWords = tokenizeWords(corrected);

    const diffs = [];
    const len = Math.min(oWords.length, cWords.length);

    for (let i = 0; i < len; i++) {
      if (oWords[i].word !== cWords[i].word) {
        diffs.push({
          start: oWords[i].start,
          end: oWords[i].end,
          original: oWords[i].word,
          suggestion: cWords[i].word,
        });
      }
    }
    return diffs;
  }

  /* -------------------------------
     RENDER HIGHLIGHTS (UNCHANGED)
  --------------------------------*/
  function buildHighlightedHTML(text, diffs) {
    if (!diffs.length) return htmlWithNewlines(text);

    let html = "";
    let last = 0;

    for (const d of diffs) {
      html += htmlWithNewlines(text.slice(last, d.start));
      html += `<span class="error"
      data-original="${escapeHTML(d.original)}"
      data-suggestion="${escapeHTML(d.suggestion)}"
    >${escapeHTML(d.original)}</span>`;
    
          last = d.end;
    }

    html += htmlWithNewlines(text.slice(last));
    return html;
  }

  /* -------------------------------
     TOOLTIP PORTAL (NO LAYOUT SHIFT)
  --------------------------------*/
/* -------------------------------
   TOOLTIP PORTAL (REUSE EXISTING)
--------------------------------*/
const tooltip = document.querySelector(".tooltip-portal");

if (!tooltip) {
  console.error("❌ tooltip-portal not found — it must be created in index.html");
} else {
  console.log("✅ Reusing existing tooltip:", tooltip);
}

window.__tcTooltip = tooltip;


  editor.addEventListener("click", (e) => {
    console.log("🖱️ Editor clicked:", e.target);
    const error = e.target.closest(".error");
    if (!error) return;
  
    activeError = error;
    console.log("❗ Active error set:", activeError, {
      original: error.dataset.original,
      suggestion: error.dataset.suggestion
    });
  
    // 1️⃣ Inject tooltip content
    tooltip.innerHTML = `
      <div class="suggestion">${error.dataset.suggestion}</div>
      <div class="actions">
        <button type="button" class="apply">Tillämpa</button>
        <button type="button" class="dismiss">Avvisa</button>
      </div>
    `;
  
    // 2️⃣ Force render (hidden) so dimensions are REAL
    tooltip.style.visibility = "hidden";
    tooltip.style.display = "block";
    tooltip.style.left = "0px";
    tooltip.style.top = "0px";
    tooltip.style.transform = "none";
  
    // 3️⃣ Measure
    const wordRect = error.getBoundingClientRect();
    const tipRect  = tooltip.getBoundingClientRect();
  
    const GAP = 12;
    const PADDING = 8;
  
    // 4️⃣ Position ABOVE word (fallback BELOW if needed)
    let top = wordRect.top - tipRect.height - GAP;
    if (top < PADDING) {
      top = wordRect.bottom + GAP;
    }
  
    // 5️⃣ Center horizontally + clamp to viewport
    let left = wordRect.left + wordRect.width / 2;
    const minLeft = PADDING + tipRect.width / 2;
    const maxLeft = window.innerWidth - PADDING - tipRect.width / 2;
    left = Math.max(minLeft, Math.min(maxLeft, left));
  
    // 6️⃣ Apply final position & show
    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${left}px`;
    tooltip.style.transform = "translateX(-50%)";
    tooltip.style.visibility = "visible";
    tooltip.classList.add("visible");
  });

  /* -------------------------------
     APPLY / DISMISS (WORKING)
  --------------------------------*/
document.addEventListener("click", (e) => {
  const applyBtn = e.target.closest(".tooltip-portal .apply");
  const dismissBtn = e.target.closest(".tooltip-portal .dismiss");
  if (!applyBtn && !dismissBtn) return;

  e.preventDefault();
  e.stopPropagation();

  if (!activeError) return;

  if (applyBtn) {
    const suggestion = activeError.dataset.suggestion || "";
    activeError.replaceWith(document.createTextNode(suggestion));
  } else {
    const original = activeError.dataset.original || "";
    activeError.replaceWith(document.createTextNode(original));
  }

  lastPlainText = getPlainText();
  sessionStorage.setItem("tc_text", lastPlainText);
  updateCounts(lastPlainText);

  document.querySelector(".tooltip-portal")?.classList.remove("visible");
  activeError = null;
}, true);

  
  /* -------------------------------
     INIT
  /* -------------------------------
     INIT
  --------------------------------*/
  /* -------------------------------
     INIT
  --------------------------------*/
  const saved = sessionStorage.getItem("tc_text");
  if (saved) {
    lastPlainText = saved;
    editor.innerHTML = htmlWithNewlines(saved);
  }

  updateCounts(lastPlainText);
  editor.focus();

  /* -------------------------------
     INPUT
  --------------------------------*/
  editor.addEventListener("input", () => {
    lastPlainText = getPlainText();
    sessionStorage.setItem("tc_text", lastPlainText);
    updateCounts(lastPlainText);
  
    enforceLengthLimit(lastPlainText);
  });
  
  

  clearBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    editor.innerHTML = "";
    lastPlainText = "";
    sessionStorage.removeItem("tc_text");
    updateCounts("");
    editor.focus();
  });

  /* -------------------------------
     SUBMIT
  --------------------------------*/
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
  
    const text = lastPlainText.trim();
    if (!text) return;
  
    // 🚫 HARD STOP — NEVER allow loading when too long
    const wordCount = countWords(text);

    if (wordCount > MAX_WORDS) {
      enforceLengthLimit(text);
      submit.classList.remove("is-loading");
      return;
    }
    
  
    // ✅ Only now we allow loading
    submit.disabled = true;
    submit.classList.add("is-loading");
  
    try {
      const res = await fetch("/", {
        method: "POST",
        headers: {
          "X-CSRFToken": document.querySelector("[name=csrfmiddlewaretoken]").value,
          "X-Requested-With": "XMLHttpRequest",
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({ text }),
      });
  
      // 🔐 NOT LOGGED IN
      if (res.status === 401) {
        submit.disabled = false;
        submit.classList.remove("is-loading");
  
        if (window.openRegisterModal) {
          window.openRegisterModal();
        } else if (window.openAuthModal) {
          window.openAuthModal();
        }
        return;
      }
  
      const data = await res.json();
      const diffs = getWordDiffs(
        data.original_text,
        data.corrected_text
      );
  
      editor.innerHTML = buildHighlightedHTML(
        data.original_text,
        diffs
      );
  
      lastPlainText = data.original_text;
      sessionStorage.setItem("tc_text", lastPlainText);
      updateCounts(lastPlainText);
  
    } catch (err) {
      console.error("Submit error:", err);
    } finally {
      submit.disabled = false;
      submit.classList.remove("is-loading");
    }
  });
  


})();
