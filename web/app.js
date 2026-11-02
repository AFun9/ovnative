const $ = (id) => document.getElementById(id);

const SAMPLES = {
  zh: "今天天气不错，我们一起去爬爬山吧",
  en: "The weather is lovely today. Let's go for a hike together.",
};

const I18N = {
  zh: {
    title: "OVNative · 本地语音合成",
    statusConnecting: "正在连接本地引擎…",
    statusCuda: "本地 CUDA 引擎就绪",
    statusCpu: "本地 CPU 引擎就绪",
    statusBad: "无法连接本地引擎",
    heroTitle: "把文字，变成声音。",
    heroLede: "支持 600+ 语种、零样本音色克隆、用文字描述音色，以及自然、有韵律的语音合成。",
    featLanguages: "600+ 语种",
    featClone: "零样本克隆",
    featDesign: "文字设计音色",
    featProsody: "自然韵律",
    textLabel: "要说的文字",
    languageLabel: "语言",
    stepsLabel: "步数",
    seedLabel: "Seed",
    cfgLabel: "CFG",
    instructLabel: "音色 / 风格（可选）",
    instructPlaceholder: "male, british accent",
    autoOption: "自动",
    advancedSummary: "Voice clone（参考音频）",
    dropTitle: "拖入或选择一段 WAV 参考音频",
    dropSub: "24 kHz / 单声道最好；会自动做静音裁剪与 RMS 归一化",
    refTextLabel: "参考音频转写（可选，但推荐）",
    refTextPlaceholder: "希望你以后过的比我还好哟！",
    generate: "生成语音",
    generating: "生成中…",
    clear: "清空",
    hint: "首次生成会加载权重，之后是秒级。",
    wait: "正在本机推理，请稍候…",
    uploading: "正在上传参考音频…",
    pending: "待上传 · ",
    refReady: "已载入参考音频 · ",
    outputEyebrow: "OUTPUT",
    outputTitle: "生成结果",
    statDuration: "时长",
    statTime: "生成耗时",
    statRtf: "RTF",
    download: "下载 WAV",
    empty: "还没有生成。左边写好句子，点“生成语音”。",
    footerLeft: "OVNative · native TTS runtime",
    footerRight: "所有计算都在本机完成",
    failed: "失败：",
    unknownError: "未知错误",
  },
  en: {
    title: "OVNative · Local TTS",
    statusConnecting: "Connecting to the local engine…",
    statusCuda: "Local CUDA engine ready",
    statusCpu: "Local CPU engine ready",
    statusBad: "Cannot reach the local engine",
    heroTitle: "Turn text into voice.",
    heroLede: "600+ languages, zero-shot voice cloning, voice design from a text description, and natural prosody.",
    featLanguages: "600+ languages",
    featClone: "Zero-shot clone",
    featDesign: "Voice design",
    featProsody: "Natural prosody",
    textLabel: "Text to speak",
    languageLabel: "Language",
    stepsLabel: "Steps",
    seedLabel: "Seed",
    cfgLabel: "CFG",
    instructLabel: "Voice / style (optional)",
    instructPlaceholder: "male, british accent",
    autoOption: "Auto",
    advancedSummary: "Voice clone (reference audio)",
    dropTitle: "Drop or choose a reference WAV",
    dropSub: "24 kHz mono works best; silence trimming and RMS normalisation are automatic",
    refTextLabel: "Reference transcript (optional, recommended)",
    refTextPlaceholder: "Transcript of the reference audio",
    generate: "Generate speech",
    generating: "Generating…",
    clear: "Clear",
    hint: "The first run loads the weights; later runs are fast.",
    wait: "Running locally. Please wait…",
    uploading: "Uploading reference audio…",
    pending: "Ready to upload · ",
    refReady: "Reference loaded · ",
    outputEyebrow: "OUTPUT",
    outputTitle: "Result",
    statDuration: "Duration",
    statTime: "Elapsed",
    statRtf: "RTF",
    download: "Download WAV",
    empty: "Nothing generated yet. Write a sentence on the left, then click Generate.",
    footerLeft: "OVNative · native TTS runtime",
    footerRight: "Everything runs on this machine",
    failed: "Failed: ",
    unknownError: "Unknown error",
  },
};

const state = {
  refId: null,
  refFile: null,
  audioUrl: null,
  busy: false,
  lang: "zh",
  engine: null,
  error: null,
};

function t() { return I18N[state.lang]; }

function applyStatus() {
  const el = $("status");
  const text = $("status-text");
  el.classList.remove("ok", "bad");
  if (state.engine === "cuda" || state.engine === "cpu") {
    el.classList.add("ok");
    text.textContent = state.engine === "cuda" ? t().statusCuda : t().statusCpu;
    $("backend-badge").textContent = state.engine === "cuda" ? "CUDA" : "CPU";
  } else if (state.engine === "error") {
    el.classList.add("bad");
    text.textContent = t().statusBad;
    $("backend-badge").textContent = "—";
  } else {
    text.textContent = t().statusConnecting;
  }
}

function applyLang(lang, swapSample = true) {
  const old = state.lang;
  state.lang = lang === "en" ? "en" : "zh";
  localStorage.setItem("ovnative.lang", state.lang);
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.title = t().title;
  const d = t();
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (d[key]) el.textContent = d[key];
  });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => {
    const key = el.getAttribute("data-i18n-ph");
    if (d[key]) el.placeholder = d[key];
  });
  $("lang-toggle").textContent = state.lang === "zh" ? "EN" : "中文";
  if (swapSample && old !== state.lang) {
    const ta = $("text");
    if (ta.value === SAMPLES[old] || ta.value === "") ta.value = SAMPLES[state.lang];
  }
  if (state.refFile) {
    $("ref-status").textContent = (state.refId ? t().refReady : t().pending) + state.refFile.name;
  } else {
    $("ref-status").textContent = t().dropSub;
  }
  if (state.busy) $("hint").textContent = t().wait;
  else if (!state.error) $("hint").textContent = t().hint;
  applyStatus();
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
  btn.querySelector(".btn-label").textContent = on ? t().generating : t().generate;
  if (on) {
    state.error = null;
    $("hint").textContent = t().wait;
  } else if (!state.error) {
    $("hint").textContent = t().hint;
  }
}

async function health() {
  state.engine = null;
  applyStatus();
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    const j = await r.json();
    state.engine = j.cuda ? "cuda" : "cpu";
  } catch {
    state.engine = "error";
  }
  applyStatus();
}

async function uploadRef(file) {
  const status = $("ref-status");
  status.textContent = t().uploading;
  const r = await fetch("/api/ref", {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: file,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || t().unknownError);
  state.refId = j.id;
  status.textContent = t().refReady + file.name;
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
    state.error = e;
    $("hint").textContent = t().failed + e.message;
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
    $("ref-status").textContent = f ? t().pending + f.name : t().dropSub;
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
    $("ref-status").textContent = t().pending + f.name;
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
  state.error = null;
  $("ref-status").textContent = t().dropSub;
  $("hint").textContent = t().hint;
  $("text").focus();
});
$("lang-toggle").addEventListener("click", () => applyLang(state.lang === "zh" ? "en" : "zh"));

const saved = localStorage.getItem("ovnative.lang");
const initial = saved || ((navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en");
applyLang(initial, false);
if (initial === "en") {
  $("text").value = SAMPLES.en;
  $("language").value = "English";
}
setupDrop();
health();
