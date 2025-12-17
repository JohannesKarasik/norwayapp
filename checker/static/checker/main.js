(function () {
  console.log("✅ main.js loaded");

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
    const regex = /\p{L}+/gu;
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
      html += `<span class="error" data-original="${escapeHTML(d.original)}" data-suggestion="${escapeHTML(d.suggestion)}">${escapeHTML(d.original)}</span>`;
      last = d.end;
    }

    html += htmlWithNewlines(text.slice(last));
    return html;
  }

  /* -------------------------------
     TOOLTIP PORTAL (NO LAYOUT SHIFT)
  --------------------------------*/
  const tooltip = document.createElement("div");
  tooltip.className = "tooltip-portal";
  document.body.appendChild(tooltip);

  editor.addEventListener("click", (e) => {
    const error = e.target.closest(".error");
    if (!error) return;

    activeError = error;

    const rect = error.getBoundingClientRect();

    tooltip.innerHTML = `
    <div class="suggestion">${error.dataset.suggestion}</div>
    <div class="actions">
      <button class="apply">Tillämpa</button>
      <button class="dismiss">Avvisa</button>
    </div>
  `;
  

  const tooltipTop =
  rect.top + window.scrollY + rect.height / 2;

const tooltipLeft =
  rect.left + window.scrollX + rect.width / 2;

tooltip.style.top = `${tooltipTop}px`;
tooltip.style.left = `${tooltipLeft}px`;
tooltip.style.transform = "translate(-50%, -50%)";

    tooltip.classList.add("visible");
  });

  tooltip.addEventListener("click", (e) => {
    if (!activeError) return;

    if (e.target.classList.contains("apply")) {
      activeError.outerHTML = escapeHTML(activeError.dataset.suggestion);
    }

    if (e.target.classList.contains("dismiss")) {
      activeError.outerHTML = escapeHTML(activeError.dataset.original);
    }

    tooltip.classList.remove("visible");
    activeError = null;

    lastPlainText = getPlainText();
    sessionStorage.setItem("tc_text", lastPlainText);
    updateCounts(lastPlainText);
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".error") && !e.target.closest(".tooltip-portal")) {
      tooltip.classList.remove("visible");
      activeError = null;
    }
  });

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
