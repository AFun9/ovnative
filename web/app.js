const $ = (id) => document.getElementById(id);
const state = { refId: null, refFile: null, audioUrl: null, busy: false };

async function health() {
  const el = $("status");
  const text = $("status-text");
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    const j = await r.json();
    el.classList.remove("bad");
    el.classList.add("ok");
    text.textContent = j.cuda ? "本地 CUDA 引擎就绪" : "本地 CPU 引擎就绪";
    $("backend-badge").textContent = j.cuda ? "CUDA" : "CPU";
  } catch {
    el.classList.remove("ok");
    el.classList.add("bad");
    text.textContent = "无法连接本地引擎";
  }
}

function fmtDuration(sec) {
  if (!Number.isFinite(sec)) return "—";
  return sec.toFixed(2) + " s";
}

function setLoading(on) {
  state.busy = on;
  const btn = $("generate");
  btn.disabled = on;
  btn.classList.toggle("loading", on);
  btn.querySelector(".btn-label").textContent = on ? "生成中…" : "生成语音";
  $("hint").textContent = on ? "正在本机推理，请稍候…" : "首次生成会加载权重，之后是秒级。";
}

async function uploadRef(file) {
  const status = $("ref-status");
  status.textContent = "正在上传参考音频…";
  const r = await fetch("/api/ref", {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: file,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || "参考音频上传失败");
  state.refId = j.id;
  status.textContent = `已载入参考音频 · ${file.name}`;
}

async function generate(ev) {
  ev.preventDefault();
  if (state.busy) return;
  const text = $("text").value.trim();
  if (!text) {
    $("text").focus();
    return;
  }
  setLoading(true);
  try {
    if (state.refFile && !state.refId) await uploadRef(state.refFile);
    const body = new URLSearchParams({
      text,
      language: $("language").value,
      instruct: $("instruct").value.trim(),
      steps: $("steps").value || "32",
      cfg: $("cfg").value || "2.0",
      seed: $("seed").value || "1234",
    });
    if (state.refId) {
      body.set("ref_id", state.refId);
      body.set("ref_text", $("ref-text").value.trim());
    }
    const t0 = performance.now();
    const r = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
      body,
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.error || `HTTP ${r.status}`);
    }
    const blob = await r.blob();
    const dt = (performance.now() - t0) / 1000;
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = URL.createObjectURL(blob);
    const audio = $("audio");
    audio.src = state.audioUrl;
    audio.load();
    const download = $("download");
    download.href = state.audioUrl;
    download.download = `ovnative-${Date.now()}.wav`;
    download.classList.remove("disabled");
    $("empty").style.display = "none";
    $("stat-time").textContent = dt.toFixed(2) + " s";
    audio.onloadedmetadata = () => {
      const dur = audio.duration;
      $("stat-duration").textContent = fmtDuration(dur);
      $("stat-rtf").textContent = Number.isFinite(dur) && dur > 0 ? (dt / dur).toFixed(2) : "—";
    };
    audio.play().catch(() => {});
  } catch (e) {
    $("hint").textContent = "失败：" + e.message;
  } finally {
    setLoading(false);
  }
}

function setupDrop() {
  const drop = $("drop");
  const input = $("ref-file");
  input.addEventListener("change", () => {
    const f = input.files && input.files[0];
    state.refFile = f || null;
    state.refId = null;
    $("ref-status").textContent = f ? `待上传 · ${f.name}` : "24 kHz / 单声道最好；会自动做静音裁剪与 RMS 归一化";
  });
  ["dragenter", "dragover"].forEach((n) => drop.addEventListener(n, (e) => {
    e.preventDefault();
    drop.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((n) => drop.addEventListener(n, (e) => {
    e.preventDefault();
    drop.classList.remove("drag");
  }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    input.files = e.dataTransfer.files;
    state.refFile = f;
    state.refId = null;
    $("ref-status").textContent = `待上传 · ${f.name}`;
  });
}

$("form").addEventListener("submit", generate);
$("clear").addEventListener("click", () => {
  $("text").value = "";
  $("instruct").value = "";
  $("ref-text").value = "";
  $("ref-file").value = "";
  state.refFile = null;
  state.refId = null;
  $("ref-status").textContent = "24 kHz / 单声道最好；会自动做静音裁剪与 RMS 归一化";
  $("text").focus();
});
setupDrop();
health();
