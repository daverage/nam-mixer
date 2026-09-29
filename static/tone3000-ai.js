// TONE3000 tab, "Describe a tone (AI)" mode: tone brief -> AI-ranked packs -> per-pack files and questions.
// Uses app.js globals (beginActivity, announce, triggerFileDownload, desktopSaveLabel), so it loads after app.js.
(() => {
  const section = document.getElementById("tone3000-ai-section");
  if (!section) return;
  const standard = document.getElementById("tone3000-standard-section");
  const thread = document.getElementById("t3ai-thread");
  const promptEl = document.getElementById("tone3000-ai-prompt");
  const label = document.getElementById("t3ai-label");
  const research = document.getElementById("tone3000-ai-research");
  const rigScope = document.getElementById("tone3000-ai-rig-scope");
  const author = document.getElementById("tone3000-ai-author");
  const status = document.getElementById("tone3000-ai-status");
  const grid = document.getElementById("tone3000-ai-results");
  const searchBtn = document.getElementById("btn-tone3000-ai-search");
  const resetBtn = document.getElementById("btn-t3ai-reset");

  const state = { history: [], goal: "", busy: false };
  const GEAR_LABELS = { amp: "Amp head", "amp-cab": "Full rig", amp_cab: "Full rig", "full-rig": "Full rig", pedal: "Pedal", outboard: "Outboard", ir: "IR" };
  const KIND_LABELS = { amp: "Amps", effect: "Effects", guitar: "Guitars", pickup: "Pickups", cab: "Cabs", other: "Other" };

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  };
  const compact = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n));
  const post = async (url, body) => {
    const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  };

  document.querySelectorAll("input[name='tone3000-mode']").forEach((radio) => {
    radio.addEventListener("change", () => {
      const ai = radio.checked && radio.value === "ai";
      standard.hidden = ai;
      section.hidden = !ai;
      if (ai) promptEl.focus();
    });
  });

  function setBusy(busy) {
    state.busy = busy;
    searchBtn.disabled = busy;
    promptEl.disabled = busy;
  }

  function renderBrief(prompt, data) {
    const turn = el("div", "t3ai-turn");
    turn.append(el("div", "t3ai-you", prompt));
    const brief = el("article", "t3ai-brief");
    const plan = data.plan;
    brief.append(el("h3", null, "Tone brief"), el("p", "t3ai-summary", plan.summary));
    if (plan.gear.length) {
      const groups = {};
      plan.gear.forEach((g) => { (groups[g.kind] ||= []).push(g); });
      const gear = el("div", "t3ai-gear");
      Object.keys(KIND_LABELS).filter((k) => groups[k]).forEach((kind) => {
        const row = el("div", "t3ai-gear-row");
        row.append(el("span", "t3ai-gear-kind", KIND_LABELS[kind]));
        groups[kind].forEach((g) => {
          const chip = el("span", `t3ai-chip t3ai-chip-${kind}`, g.name);
          if (g.role) chip.title = g.role;
          row.append(chip);
        });
        gear.append(row);
      });
      brief.append(gear);
    }
    if (plan.advice.length) {
      const details = el("details", "t3ai-advice");
      details.open = state.history.length === 0;
      details.append(el("summary", null, "How to get there"));
      const list = el("ul");
      plan.advice.forEach((tip) => list.append(el("li", null, tip)));
      details.append(list);
      brief.append(details);
    }
    const meta = el("p", "t3ai-meta", `Searched TONE3000 for ${data.queries.map((q) => `"${q}"`).join(", ")}${data.researched ? " after web research" : ""}.`);
    brief.append(meta);
    data.warnings.forEach((w) => brief.append(el("p", "t3ai-warning", w)));
    turn.append(brief);
    thread.append(turn);
    turn.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function packCard(pack) {
    const card = el("article", "t3ai-card");
    const media = el("div", "t3ai-media");
    if (pack.image) {
      const img = document.createElement("img");
      img.src = pack.image; img.alt = ""; img.loading = "lazy"; img.referrerPolicy = "no-referrer";
      img.addEventListener("error", () => { img.remove(); media.classList.add("t3ai-media-empty"); });
      media.append(img);
    } else media.classList.add("t3ai-media-empty");
    if (Number.isFinite(pack.ai_fit)) {
      const fit = el("span", "t3ai-fit", `${pack.ai_fit}% fit`);
      fit.dataset.level = pack.ai_fit >= 75 ? "high" : pack.ai_fit >= 50 ? "mid" : "low";
      media.append(fit);
    }
    const body = el("div", "t3ai-card-body");
    body.append(el("h3", null, pack.title), el("p", "t3ai-by", `by ${pack.creator}`));
    const facts = el("div", "t3ai-facts");
    if (pack.gear) facts.append(el("span", "t3ai-fact", GEAR_LABELS[pack.gear] || pack.gear));
    if (Number.isFinite(pack.a2_models_count)) facts.append(el("span", "t3ai-fact", `${pack.a2_models_count} A2 NAM${pack.a2_models_count === 1 ? "" : "s"}`));
    if (Number.isFinite(pack.downloads_count)) facts.append(el("span", "t3ai-fact", `${compact(pack.downloads_count)} downloads`));
    body.append(facts);
    if (pack.ai_why) body.append(el("p", "t3ai-why", pack.ai_why));
    if (pack.tags && pack.tags.length) {
      const tags = el("div", "t3ai-tags");
      pack.tags.slice(0, 6).forEach((t) => tags.append(el("span", "t3ai-tag", t)));
      body.append(tags);
    }
    if (pack.description) {
      const d = el("details", "t3ai-desc");
      d.append(el("summary", null, "Pack description"), el("p", null, pack.description));
      body.append(d);
    }
    const actions = el("div", "t3ai-actions");
    const open = el("button", "btn btn-primary btn-small", "Files & questions");
    open.type = "button";
    actions.append(open);
    if (pack.url) {
      const link = el("a", "btn btn-secondary btn-small", "Open on TONE3000");
      link.href = pack.url; link.target = "_blank"; link.rel = "noopener noreferrer";
      actions.append(link);
    }
    body.append(actions);
    const panel = el("div", "t3ai-panel");
    panel.hidden = true;
    body.append(panel);
    card.append(media, body);
    open.addEventListener("click", () => togglePanel(card, panel, open, pack));
    return card;
  }

  async function togglePanel(card, panel, button, pack) {
    if (!panel.hidden) {
      panel.hidden = true; card.classList.remove("is-open"); button.textContent = "Files & questions";
      return;
    }
    panel.hidden = false; card.classList.add("is-open"); button.textContent = "Close";
    if (panel.dataset.loaded) return;
    panel.replaceChildren(el("p", "info", "Loading the NAM files in this pack..."));
    try {
      const response = await fetch(`/api/tone3000/tones/${encodeURIComponent(pack.id)}/models`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not load this pack");
      buildPanel(panel, pack, data.models);
      panel.dataset.loaded = "1";
    } catch (error) {
      panel.replaceChildren(el("p", "t3ai-warning", error.message));
    }
  }

  function buildPanel(panel, pack, models) {
    panel.replaceChildren();
    const files = el("ul", "t3ai-files");
    const rows = new Map();
    models.forEach((model) => {
      const row = el("li", "t3ai-file");
      const name = el("span", "t3ai-file-name", model.name);
      const download = el("button", "btn btn-secondary btn-small", desktopSaveLabel("Download"));
      download.type = "button";
      download.setAttribute("aria-label", desktopSaveLabel(`Download ${model.name}`));
      const url = `/api/tone3000/tones/${encodeURIComponent(pack.id)}/models/${encodeURIComponent(model.id)}/download`;
      download.addEventListener("click", () => triggerFileDownload(url, model.name));
      row.append(name, download);
      rows.set(model.name, row);
      files.append(row);
    });
    const fileColumn = el("div", "t3ai-file-column");
    fileColumn.append(el("h4", null, `${models.length} NAM file${models.length === 1 ? "" : "s"}`), files);
    panel.append(fileColumn);

    const chat = el("div", "t3ai-chat");
    const log = el("div", "t3ai-chat-log");
    const input = el("textarea", "t3ai-chat-input");
    input.rows = 2; input.maxLength = 600;
    input.placeholder = "Ask about this pack, e.g. which file suits rhythm, what the names mean, how it compares";
    input.setAttribute("aria-label", `Ask about ${pack.title}`);
    const ask = el("button", "btn btn-primary btn-small", "Ask");
    ask.type = "button";
    const suggestions = el("div", "t3ai-suggest");
    ["Which file fits my tone best?", "What do the file names mean?", "Which is best for rhythm vs lead?"].forEach((q) => {
      const s = el("button", "link-btn", q);
      s.type = "button";
      s.addEventListener("click", () => { input.value = q; send(); });
      suggestions.append(s);
    });
    const history = [];
    async function send() {
      const question = input.value.trim();
      if (!question || ask.disabled) return;
      ask.disabled = true; input.disabled = true;
      log.append(el("div", "t3ai-you", question));
      const thinking = el("div", "t3ai-ai info", "Thinking...");
      log.append(thinking);
      input.value = "";
      const stop = beginActivity(`Asking about ${pack.title}...`);
      try {
        const data = await post("/api/tone3000/ai_pack_chat", {
          tone_id: pack.id, question, tone_goal: state.goal, history,
          pack: { title: pack.title, creator: pack.creator, description: pack.description, tags: pack.tags || [] },
        });
        thinking.className = "t3ai-ai"; thinking.textContent = data.reply;
        history.push({ role: "user", content: question }, { role: "assistant", content: data.reply });
        rows.forEach((row) => row.classList.remove("is-pick"));
        data.recommended_files.forEach((name) => {
          const row = rows.get(name);
          if (!row) return;
          row.classList.add("is-pick");
          if (!row.querySelector(".t3ai-pick")) row.querySelector(".t3ai-file-name").append(" ", el("span", "t3ai-pick", "AI pick"));
        });
        announce(data.reply);
      } catch (error) {
        thinking.className = "t3ai-warning"; thinking.textContent = error.message;
      } finally {
        stop(); ask.disabled = false; input.disabled = false; input.focus();
      }
    }
    ask.addEventListener("click", send);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
    const composer = el("div", "t3ai-chat-composer");
    composer.append(input, ask);
    chat.append(el("h4", null, "Ask about this pack"), suggestions, log, composer);
    panel.append(chat);
  }

  async function search() {
    const prompt = promptEl.value.trim();
    if (!prompt || state.busy) { if (!prompt) status.textContent = "Describe the tone you want first."; return; }
    setBusy(true);
    status.textContent = research.checked ? "Researching the tone, then searching TONE3000... this can take a minute." : "Working out the tone, then searching TONE3000...";
    const stop = beginActivity("Finding tones on TONE3000...");
    try {
      const data = await post("/api/tone3000/ai_search", {
        prompt, use_research: research.checked, rig_scope: rigScope.value, author: author.value.trim(), history: state.history,
      });
      if (!state.goal) state.goal = prompt;
      else state.goal = `${state.goal} / refined: ${prompt}`.slice(-600);
      state.history.push({ role: "user", content: prompt }, { role: "assistant", content: data.plan.summary });
      state.history = state.history.slice(-12);
      renderBrief(prompt, data);
      const rated = data.results.filter((p) => Number.isFinite(p.ai_fit));
      const others = data.results.filter((p) => !Number.isFinite(p.ai_fit));
      grid.replaceChildren(...rated.map(packCard));
      if (others.length) {
        grid.append(el("h3", "t3ai-grid-heading", rated.length ? "Other catalogue matches (not rated by the AI)" : "Catalogue matches"));
        grid.append(...others.map(packCard));
      }
      grid.hidden = data.results.length === 0;
      status.textContent = data.results.length
        ? `${rated.length ? `${rated.length} rated by the AI, best fit first` : "Showing catalogue order"}; ${data.results.length} pack${data.results.length === 1 ? "" : "s"} in total.`
        : "No TONE3000 packs matched. Try naming an amp, or set Rigs to Anything.";
      announce(`${data.plan.summary} ${status.textContent}`);
      promptEl.value = "";
      label.textContent = "Refine the search (e.g. more gain, darker, a different era, a cheaper amp)";
      promptEl.placeholder = "e.g. a bit more gain for solos";
      resetBtn.hidden = false;
    } catch (error) {
      status.textContent = error.message;
    } finally {
      stop(); setBusy(false);
    }
  }

  searchBtn.addEventListener("click", search);
  promptEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); search(); } });
  resetBtn.addEventListener("click", () => {
    state.history = []; state.goal = "";
    thread.replaceChildren(); grid.replaceChildren(); grid.hidden = true;
    status.textContent = ""; resetBtn.hidden = true;
    label.textContent = "Describe the tone you want";
    promptEl.placeholder = "e.g. Early Stevie Ray Vaughan: fat, bright, on the edge of breakup, digs in when I pick hard";
    promptEl.focus();
  });
})();
