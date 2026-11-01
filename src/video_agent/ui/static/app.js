// Video Editor Agent UI: no framework, no build step.
const $ = (id) => document.getElementById(id);
const els = {
  model: $("model-chip"), shotList: $("shot-list"), shotCount: $("shot-count"), filter: $("filter"),
  player: $("player"), playerEmpty: $("player-empty"), timeline: $("timeline"), ruler: $("ruler"),
  duration: $("timeline-duration"), messages: $("messages"), welcome: $("welcome"),
  composer: $("composer"), prompt: $("prompt"), send: $("send-btn"), status: $("agent-status"),
  render: $("render-btn"), reset: $("reset-btn"), toast: $("toast"),
};

let state = { shots: [], timeline: [], preview_url: null };
let busy = false;

const icons = {
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
  left: '<svg viewBox="0 0 24 24"><path d="M15 6l-6 6 6 6" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  right: '<svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  close: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};

// ---------- helpers ----------
const escape = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (s) => {
  const m = Math.floor(s / 60), sec = s - m * 60;
  return `${m}:${sec.toFixed(1).padStart(4, "0")}`;
};
// Tiny markdown: **bold**, `code`. Everything else stays plain text (white-space: pre-wrap).
const md = (text) => escape(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");

function toast(message, kind = "error") {
  els.toast.textContent = message;
  els.toast.className = `toast ${kind === "info" ? "info" : ""}`;
  els.toast.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (els.toast.hidden = true), 4500);
}

async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function apply(next) {
  state = { ...state, ...next };
  if (next.model) els.model.textContent = next.model;
  if (next.shots) renderShots();
  if (next.timeline) renderTimeline();
  if ("preview_url" in next) renderPreview();
}

// ---------- shots ----------
function renderShots() {
  const q = els.filter.value.trim().toLowerCase();
  const shots = state.shots.filter((s) =>
    !q || [s.shot_id, s.asset_id, s.transcript, ...(s.captions || [])].join(" ").toLowerCase().includes(q));
  els.shotCount.textContent = q ? `${shots.length} / ${state.shots.length}` : `${state.shots.length}`;
  els.shotList.innerHTML = shots.length ? "" : '<p class="empty-note">No shot matches this filter.</p>';
  for (const s of shots) {
    const card = document.createElement("article");
    card.className = "shot";
    card.dataset.shot = s.shot_id;
    card.innerHTML = `
      <div class="thumb">
        ${s.keyframe_url ? `<img src="${s.keyframe_url}" alt="" loading="lazy">` : ""}
        <span class="tc">${fmt(s.start)} → ${fmt(s.end)}</span>
        <button class="add" aria-label="Add ${escape(s.shot_id)} to the timeline" title="Add to timeline">${icons.plus}</button>
      </div>
      <div class="shot-body">
        <span class="shot-id">${escape(s.shot_id)}</span>
        ${s.captions?.length ? `<span class="caption">${escape(s.captions.join("; "))}</span>` : ""}
        ${s.transcript ? `<span class="speech">${escape(s.transcript)}</span>` : ""}
      </div>`;
    card.querySelector(".add").addEventListener("click", () => edit("/api/timeline/insert", { shot_id: s.shot_id }));
    els.shotList.append(card);
  }
}

function flashShots(ids) {
  for (const id of ids) {
    const card = els.shotList.querySelector(`[data-shot="${CSS.escape(id)}"]`);
    if (!card) continue;
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 2200);
  }
}

// ---------- timeline ----------
function renderTimeline() {
  const clips = state.timeline || [];
  const total = clips.length ? clips[clips.length - 1].timeline_end : 0;
  els.duration.textContent = clips.length ? `${clips.length} clip${clips.length > 1 ? "s" : ""} · ${fmt(total)}` : "";
  const pxPerSec = 22;
  const thumbs = Object.fromEntries(state.shots.map((s) => [s.shot_id, s.keyframe_url]));

  els.timeline.innerHTML = "";
  clips.forEach((c, i) => {
    const li = document.createElement("li");
    li.className = "clip";
    const duration = c.timeline_end - c.timeline_start;
    li.style.width = `${Math.max(64, duration * pxPerSec)}px`;
    const thumb = thumbs[c.shot_id];
    if (thumb) li.style.backgroundImage = `url("${thumb}")`;
    li.title = `${c.shot_id}\nsource ${fmt(c.source_start)} → ${fmt(c.source_end)}`;
    li.innerHTML = `
      <span class="index">${i + 1}</span>
      <div class="tools">
        <button data-act="left" aria-label="Move clip ${i + 1} left" ${i === 0 ? "disabled" : ""}>${icons.left}</button>
        <button data-act="right" aria-label="Move clip ${i + 1} right" ${i === clips.length - 1 ? "disabled" : ""}>${icons.right}</button>
        <button data-act="remove" aria-label="Remove clip ${i + 1}">${icons.close}</button>
      </div>
      <div class="meta"><span class="name">${escape(c.shot_id)}</span><span class="dur">${duration.toFixed(1)} s</span></div>`;
    li.querySelector('[data-act="left"]').addEventListener("click", () => edit("/api/timeline/move", { clip_id: c.clip_id, new_index: i - 1 }));
    li.querySelector('[data-act="right"]').addEventListener("click", () => edit("/api/timeline/move", { clip_id: c.clip_id, new_index: i + 1 }));
    li.querySelector('[data-act="remove"]').addEventListener("click", () => edit("/api/timeline/remove", { clip_id: c.clip_id }));
    els.timeline.append(li);
  });

  // Ruler with a tick every 5 or 10 seconds, aligned with the clip widths (3px gap per clip).
  els.ruler.innerHTML = "";
  if (!total) return;
  const step = total > 60 ? 10 : 5;
  for (let t = 0; t <= total + 0.01; t += step) {
    const before = clips.filter((c) => c.timeline_end <= t).length;
    const mark = document.createElement("span");
    mark.style.left = `${t * pxPerSec + before * 3}px`;
    mark.textContent = fmt(t).replace(/\.0$/, "");
    els.ruler.append(mark);
  }
}

function renderPreview() {
  const url = state.preview_url;
  els.player.hidden = !url;
  els.playerEmpty.hidden = !!url;
  if (url && els.player.dataset.url !== url) {
    els.player.dataset.url = url;
    els.player.poster = state.poster_url || "";
    els.player.src = url;
    els.player.load();
  }
}

async function edit(path, body) {
  try {
    apply(await api(path, body));
    return true;
  } catch (err) {
    toast(err.message);
    return false;
  }
}

// ---------- chat ----------
function addMessage(kind, html) {
  els.welcome?.remove();
  const div = document.createElement("div");
  div.className = `msg ${kind}`;
  div.innerHTML = html;
  els.messages.append(div);
  scrollChat();
  return div;
}

function scrollChat() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function argsSummary(args) {
  return Object.entries(args || {}).map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : JSON.stringify(v)}`).join(", ");
}

function setBusy(value) {
  busy = value;
  els.send.disabled = value;
  els.render.disabled = value;
  els.reset.disabled = value;
  els.status.textContent = value ? "Working…" : "Ready";
  els.status.classList.toggle("busy", value);
}

async function send(text) {
  if (busy || !text.trim()) return;
  addMessage("user", escape(text));
  els.prompt.value = "";
  autosize();
  setBusy(true);

  let steps = null;
  const stepEls = {};
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text }),
    });
    if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, cut).replace(/^data: /, "");
        buffer = buffer.slice(cut + 2);
        if (!raw.trim()) continue;
        const ev = JSON.parse(raw);

        if (ev.type === "tool_call") {
          if (!steps) {
            steps = document.createElement("div");
            steps.className = "steps";
            els.messages.append(steps);
          }
          const step = document.createElement("div");
          step.className = "step";
          step.innerHTML = `<span class="icon"><span class="spinner"></span></span>
            <span><span class="name">${escape(ev.name)}</span><span class="args">${escape(argsSummary(ev.args))}</span></span>
            <span class="result"></span>`;
          steps.append(step);
          stepEls[ev.id] = step;
          scrollChat();
        } else if (ev.type === "tool_result") {
          const step = stepEls[ev.id];
          if (step) {
            step.classList.add(ev.ok ? "ok" : "err");
            step.querySelector(".icon").innerHTML = ev.ok ? icons.check : icons.close;
            step.querySelector(".result").textContent = ev.summary;
          }
          if (ev.ok && /insert_shot|split_clip|remove_clip|move_clip|initialize_timeline|render_preview/.test(ev.name)) {
            refresh();  // show edits live, while the agent keeps working
          }
          if (ev.ok && /search/.test(ev.name)) {
            flashShots([...ev.summary.matchAll(/"shot_id":\s*"([^"]+)"|'shot_id':\s*'([^']+)'/g)].map((m) => m[1] || m[2]));
          }
        } else if (ev.type === "message") {
          steps = null;  // the next tool calls start a new group under this message
          addMessage("agent", md(ev.content));
        } else if (ev.type === "error") {
          addMessage("error", escape(ev.message));
        } else if (ev.type === "done") {
          apply(ev.state);
          const t = document.createElement("span");
          t.className = "timing";
          t.textContent = `${ev.seconds} s`;
          els.messages.append(t);
          scrollChat();
        }
      }
    }
  } catch (err) {
    addMessage("error", escape(`Connection error: ${err.message}`));
  } finally {
    setBusy(false);
    els.prompt.focus();
  }
}

async function refresh(restoreChat = false) {
  try {
    const data = await api("/api/state");
    apply(data);
    if (restoreChat && data.conversation?.length) {
      for (const turn of data.conversation) addMessage(turn.role, turn.role === "user" ? escape(turn.content) : md(turn.content));
    }
  } catch (err) {
    toast(`Could not load the session: ${err.message}`);
  }
}

function autosize() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 160)}px`;
}

// ---------- events ----------
els.composer.addEventListener("submit", (e) => { e.preventDefault(); send(els.prompt.value); });
els.prompt.addEventListener("input", autosize);
els.prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(els.prompt.value); }
});
document.querySelectorAll("#suggestions button").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
els.filter.addEventListener("input", renderShots);
els.render.addEventListener("click", async () => {
  els.render.disabled = true;
  toast("Rendering…", "info");
  if (await edit("/api/timeline/render", {})) els.toast.hidden = true;
  els.render.disabled = busy;
});
els.reset.addEventListener("click", async () => {
  await edit("/api/reset", {});
  els.messages.innerHTML = "";
  addMessage("agent", "New edit started: the timeline and the conversation are empty.");
});

refresh(true);
