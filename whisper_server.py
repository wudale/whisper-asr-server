"""
Whisper ASR Service — faster-whisper + FastAPI
OpenAI-compatible /v1/audio/transcriptions endpoint.
CPU-only, small model, int8 quantization.

Usage:
    python3 whisper_server.py                           # start server on :9080
    python3 whisper_server.py --transcribe audio.mp3    # CLI mode, transcribe & exit
    python3 whisper_server.py --transcribe audio.mp3 --language zh --format srt
"""

import argparse
import asyncio
import json
import os
import sys
import time
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from faster_whisper import WhisperModel

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("whisper-server")

# ── Config ───────────────────────────────────────────────────────────────
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")
HOST = os.getenv("WHISPER_HOST", "0.0.0.0")
PORT = int(os.getenv("WHISPER_PORT", "9080"))

# LLM Correction Config (leave LLM_MODEL empty to disable)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# ── Globals ──────────────────────────────────────────────────────────────
model: Optional[WhisperModel] = None
model_load_time: float = 0.0


# ── Lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_load_time
    log.info(f"Loading model '{MODEL_SIZE}' (device={DEVICE}, compute={COMPUTE_TYPE})...")
    start = time.time()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    model_load_time = time.time() - start
    log.info(f"Model loaded in {model_load_time:.1f}s ✅")
    yield
    log.info("Server shutting down.")


app = FastAPI(
    title="Whisper ASR Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Demo Page ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def demo_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Whisper ASR Demo</title>
<style>
  :root { --bg:#1e1e2e; --surface:#2a2a3c; --border:#3a3a50; --text:#cdd6f4; --sub:#a6adc8; --accent:#89b4fa; --green:#a6e3a1; --red:#f38ba8; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; display:flex; justify-content:center; padding:40px 16px; }
  .container { width:100%; max-width:720px; }
  .header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:24px; flex-wrap:wrap; gap:12px; }
  .header-left h1 { font-size:24px; margin-bottom:4px; }
  .sub { color:var(--sub); font-size:14px; }
  #lang-switcher { padding:6px 10px; border-radius:6px; border:1px solid var(--border); background:var(--surface); color:var(--text); font-size:13px; cursor:pointer; }
  .dropzone { border:2px dashed var(--border); border-radius:12px; padding:40px 20px; text-align:center; cursor:pointer; transition:.2s; background:var(--surface); margin-bottom:16px; }
  .dropzone:hover,.dropzone.dragover { border-color:var(--accent); background:#2a2a40; }
  .dropzone .icon { font-size:40px; margin-bottom:8px; }
  .dropzone .hint { color:var(--sub); font-size:14px; margin-top:6px; }
  .row { display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap; }
  select, button { padding:10px 16px; border-radius:8px; border:1px solid var(--border); background:var(--surface); color:var(--text); font-size:14px; cursor:pointer; transition:.2s; }
  select:hover, button:hover { border-color:var(--accent); }
  button.primary { background:var(--accent); color:#1e1e2e; border:none; font-weight:600; flex:1; min-width:120px; }
  button.primary:hover { opacity:.9; }
  button.primary:disabled { opacity:.4; cursor:not-allowed; }
  #result { background:var(--surface); border-radius:12px; padding:20px; min-height:100px; display:none; }
  #result .lang-badge { display:inline-block; background:var(--accent); color:#1e1e2e; padding:2px 10px; border-radius:20px; font-size:13px; font-weight:600; margin-bottom:10px; }
  #result .text { font-size:16px; line-height:1.7; white-space:pre-wrap; margin-bottom:16px; }
  #result .segments { border-top:1px solid var(--border); padding-top:12px; }
  #result .seg { display:flex; gap:12px; padding:6px 0; font-size:14px; border-bottom:1px solid #2a2a3f; }
  #result .seg .ts { color:var(--sub); white-space:nowrap; font-variant-numeric:tabular-nums; min-width:100px; }
  #result .seg .txt { flex:1; }
  .err { color:var(--red); }
  .loading { text-align:center; padding:40px; color:var(--sub); }
  #file-info { font-size:13px; color:var(--sub); text-align:center; margin-bottom:12px; display:none; }
  .footer { margin-top:32px; text-align:center; color:var(--sub); font-size:13px; }
  .footer code { background:var(--surface); padding:2px 6px; border-radius:4px; font-size:12px; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="header-left">
      <h1>🎙️ Whisper ASR</h1>
      <p class="sub"><span id="status">checking...</span></p>
    </div>
    <select id="lang-switcher" onchange="switchLang(this.value)">
      <option value="en">English</option>
      <option value="zh">中文</option>
      <option value="ja">日本語</option>
      <option value="ko">한국어</option>
      <option value="de">Deutsch</option>
      <option value="fr">Français</option>
      <option value="ru">Русский</option>
      <option value="ar">العربية</option>
    </select>
  </div>

  <div class="dropzone" id="dropzone">
    <div class="icon">📁</div>
    <div data-i18n="drop_text">Drop audio file here or click to select</div>
    <div class="hint" data-i18n="drop_hint">Supports mp3, wav, m4a, ogg, flac, mp4, mov, and more</div>
  </div>
  <input type="file" id="fileInput" accept="audio/*,video/*" hidden>

  <div id="file-info"></div>

  <div class="row">
    <select id="lang">
      <option value="" data-i18n="auto_detect">Auto-detect language</option>
      <option value="zh">🇨🇳 中文</option>
      <option value="en">🇬🇧 English</option>
      <option value="ja">🇯🇵 日本語</option>
      <option value="ko">🇰🇷 한국어</option>
      <option value="fr">🇫🇷 Français</option>
      <option value="de">🇩🇪 Deutsch</option>
      <option value="es">🇪🇸 Español</option>
      <option value="ru">🇷🇺 Русский</option>
      <option value="ar">🇸🇦 العربية</option>
    </select>
    <select id="fmt">
      <option value="verbose_json" selected data-i18n="fmt_verbose">Verbose (verbose_json)</option>
      <option value="json" data-i18n="fmt_json">Plain text (json)</option>
      <option value="text" data-i18n="fmt_text">Text (text)</option>
      <option value="srt" data-i18n="fmt_srt">Subtitle (srt)</option>
    </select>
    <label style="display:flex;align-items:center;gap:6px;cursor:pointer;padding:10px 0;color:var(--text);font-size:14px;">
      <input type="checkbox" id="correctCheck" style="width:16px;height:16px;accent-color:var(--accent);cursor:pointer;">
      <span data-i18n="correct_toggle">LLM Correction</span>
    </label>
    <button class="primary" id="transcribeBtn" disabled data-i18n="btn_transcribe">Transcribe</button>
  </div>

  <div id="result"></div>

  <details open style="margin-top:24px;background:var(--surface);border-radius:12px;padding:16px;">
    <summary style="cursor:pointer;font-weight:600;font-size:15px;margin-bottom:12px;" data-i18n="api_examples">📋 API Examples</summary>
    <div class="tabs" style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
      <button class="tab active" onclick="switchTab('shell')" style="padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:var(--accent);color:#1e1e2e;cursor:pointer;font-size:13px;font-weight:600;">Shell</button>
      <button class="tab" onclick="switchTab('python')" style="padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;font-size:13px;">Python</button>
      <button class="tab" onclick="switchTab('js')" style="padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;font-size:13px;">JavaScript</button>
      <button class="tab" onclick="switchTab('java')" style="padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;font-size:13px;">Java</button>
    </div>
    <pre id="code-shell" style="background:#11111b;color:#cdd6f4;padding:14px;border-radius:8px;overflow-x:auto;font-size:13px;line-height:1.6;margin:0;"><code># Basic transcription
curl -X POST ${API_BASE}/v1/audio/transcriptions \\
  -F "file=@recording.mp3"

# Chinese + verbose output
curl -X POST ${API_BASE}/v1/audio/transcriptions \\
  -F "file=@recording.m4a" \\
  -F "language=zh" \\
  -F "response_format=verbose_json"

# Generate SRT subtitles
curl -X POST ${API_BASE}/v1/audio/transcriptions \\
  -F "file=@meeting.mp3" \\
  -F "response_format=srt"

# Health check
curl ${API_BASE}/health</code></pre>
    <pre id="code-python" style="background:#11111b;color:#cdd6f4;padding:14px;border-radius:8px;overflow-x:auto;font-size:13px;line-height:1.6;margin:0;display:none;"><code>import requests

# Basic transcription
with open("recording.mp3", "rb") as f:
    r = requests.post(
        "${API_BASE}/v1/audio/transcriptions",
        files={"file": f}
    )
print(r.json()["text"])

# Specify language + segments
with open("recording.m4a", "rb") as f:
    r = requests.post(
        "${API_BASE}/v1/audio/transcriptions",
        files={"file": f},
        data={"language": "zh", "response_format": "verbose_json"}
    )
data = r.json()
print(f"Language: {data['language']}")
for seg in data["segments"]:
    print(f"[{seg['start']:.1f}s] {seg['text']}")

# With LLM correction
with open("recording.mp3", "rb") as f:
    r = requests.post(
        "${API_BASE}/v1/audio/transcriptions",
        files={"file": f},
        data={"correct": "true", "response_format": "verbose_json"}
    )
data = r.json()
print(f"Original: {data['text']}")
print(f"Corrected: {data.get('corrected_text', '(disabled)')}")</code></pre>
    <pre id="code-js" style="background:#11111b;color:#cdd6f4;padding:14px;border-radius:8px;overflow-x:auto;font-size:13px;line-height:1.6;margin:0;display:none;"><code>// Basic transcription
const fd = new FormData();
fd.append("file", fileInput.files[0]);

const r = await fetch(
  "${API_BASE}/v1/audio/transcriptions",
  { method: "POST", body: fd }
);
const data = await r.json();
console.log(data.text);

// Specify language + verbose
fd.append("language", "zh");
fd.append("response_format", "verbose_json");

const r2 = await fetch(
  "${API_BASE}/v1/audio/transcriptions",
  { method: "POST", body: fd }
);
const { text, language, segments } = await r2.json();
segments.forEach(s => console.log(`[${s.start}s] ${s.text}`));

// With LLM correction
const fd3 = new FormData();
fd3.append("file", fileInput.files[0]);
fd3.append("correct", "true");
fd3.append("response_format", "verbose_json");
const r3 = await fetch(
  "${API_BASE}/v1/audio/transcriptions",
  { method: "POST", body: fd3 }
);
const d3 = await r3.json();
console.log("Original:", d3.text);
console.log("Corrected:", d3.corrected_text || "(disabled)");</code></pre>
    <pre id="code-java" style="background:#11111b;color:#cdd6f4;padding:14px;border-radius:8px;overflow-x:auto;font-size:13px;line-height:1.6;margin:0;display:none;"><code>// OkHttp + Okio (com.squareup.okhttp3:okhttp:4.12.0)
import okhttp3.*;
import java.io.File;
import java.io.IOException;

OkHttpClient client = new OkHttpClient();

// Basic transcription
RequestBody body = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", "recording.mp3",
        RequestBody.create(new File("recording.mp3"),
        MediaType.parse("audio/mpeg")))
    .build();

Request req = new Request.Builder()
    .url("${API_BASE}/v1/audio/transcriptions")
    .post(body)
    .build();

try (Response r = client.newCall(req).execute()) {
    System.out.println(r.body().string());
}

// Specify language
RequestBody bodyZh = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", "recording.m4a",
        RequestBody.create(new File("recording.m4a"),
        MediaType.parse("audio/mp4")))
    .addFormDataPart("language", "zh")
    .addFormDataPart("response_format", "verbose_json")
    .build();

// With LLM correction
RequestBody bodyCorrect = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", "recording.mp3",
        RequestBody.create(new File("recording.mp3"),
        MediaType.parse("audio/mpeg")))
    .addFormDataPart("correct", "true")
    .addFormDataPart("response_format", "verbose_json")
    .build();
Request reqCorrect = new Request.Builder()
    .url("${API_BASE}/v1/audio/transcriptions")
    .post(bodyCorrect)
    .build();
try (Response r = client.newCall(reqCorrect).execute()) {
    System.out.println(r.body().string());
}</code></pre>
  </details>

  <div class="footer">
    <span data-i18n="footer_prefix">Endpoint:</span> <code>POST /v1/audio/transcriptions</code> · <code>GET /health</code> · <span>Port 9080</span>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
// i18n
// ═══════════════════════════════════════════════════════════════
const I18N = {
  en: {
    drop_text: 'Drop audio file here or click to select',
    drop_hint: 'Supports mp3, wav, m4a, ogg, flac, mp4, mov, and more',
    auto_detect: 'Auto-detect language',
    fmt_verbose: 'Verbose (verbose_json)',
    fmt_json: 'Plain text (json)',
    fmt_text: 'Text (text)',
    fmt_srt: 'Subtitle (srt)',
    btn_transcribe: 'Transcribe',
    correct_toggle: 'LLM Correction',
    raw_json: 'Raw JSON',
    corrected_label: 'Corrected Text',
    api_examples: '📋 API Examples',
    footer_prefix: 'Endpoint:',
    selected: 'Selected',
    transcribing: 'Transcribing...',
    wait: 'Transcribing, please wait...',
    no_speech: '(No speech detected)',
    audio_label: 'audio',
  },
  zh: {
    drop_text: '拖拽音频文件到这里，或点击选择',
    drop_hint: '支持 mp3、wav、m4a、ogg、flac、mp4、mov 等格式',
    auto_detect: '自动检测语言',
    fmt_verbose: '详细输出 (verbose_json)',
    fmt_json: '纯文本 (json)',
    fmt_text: '纯文本 (text)',
    fmt_srt: '字幕格式 (srt)',
    btn_transcribe: '转录',
    correct_toggle: 'LLM 纠错',
    raw_json: '原始 JSON',
    corrected_label: '纠错结果',
    api_examples: '📋 API 调用示例',
    footer_prefix: '端点：',
    selected: '已选择',
    transcribing: '转录中...',
    wait: '正在转录，请稍候...',
    no_speech: '（未识别到语音）',
    audio_label: '音频',
  },
  ja: {
    drop_text: '音声ファイルをドロップ、またはクリックして選択',
    drop_hint: 'mp3, wav, m4a, ogg, flac, mp4, mov など対応',
    auto_detect: '言語を自動検出',
    fmt_verbose: '詳細 (verbose_json)',
    fmt_json: 'プレーンテキスト (json)',
    fmt_text: 'テキスト (text)',
    fmt_srt: '字幕 (srt)',
    btn_transcribe: '文字起こし',
    correct_toggle: 'LLM 修正',
    raw_json: '生JSON',
    corrected_label: '修正結果',
    api_examples: '📋 API 使用例',
    footer_prefix: 'エンドポイント：',
    selected: '選択中',
    transcribing: '文字起こし中...',
    wait: '文字起こし中、お待ちください...',
    no_speech: '（音声が検出されませんでした）',
    audio_label: '音声',
  },
  ko: {
    drop_text: '오디오 파일을 여기에 드롭하거나 클릭하여 선택',
    drop_hint: 'mp3, wav, m4a, ogg, flac, mp4, mov 등 지원',
    auto_detect: '언어 자동 감지',
    fmt_verbose: '상세 (verbose_json)',
    fmt_json: '일반 텍스트 (json)',
    fmt_text: '텍스트 (text)',
    fmt_srt: '자막 (srt)',
    btn_transcribe: '텍스트 변환',
    correct_toggle: 'LLM 보정',
    raw_json: '원본 JSON',
    corrected_label: '보정 결과',
    api_examples: '📋 API 사용 예',
    footer_prefix: '엔드포인트:',
    selected: '선택됨',
    transcribing: '변환 중...',
    wait: '변환 중입니다. 잠시만 기다려주세요...',
    no_speech: '（음성이 감지되지 않았습니다）',
    audio_label: '오디오',
  },
  de: {
    drop_text: 'Audio-Datei hier ablegen oder klicken zum Auswählen',
    drop_hint: 'Unterstützt mp3, wav, m4a, ogg, flac, mp4, mov u.a.',
    auto_detect: 'Sprache automatisch erkennen',
    fmt_verbose: 'Ausführlich (verbose_json)',
    fmt_json: 'Text (json)',
    fmt_text: 'Text (text)',
    fmt_srt: 'Untertitel (srt)',
    btn_transcribe: 'Transkribieren',
    correct_toggle: 'LLM Korrektur',
    raw_json: 'Roh-JSON',
    corrected_label: 'Korrigierter Text',
    api_examples: '📋 API-Beispiele',
    footer_prefix: 'Endpunkt:',
    selected: 'Ausgewählt',
    transcribing: 'Transkribiere...',
    wait: 'Transkribiere, bitte warten...',
    no_speech: '(Keine Sprache erkannt)',
    audio_label: 'Audio',
  },
  fr: {
    drop_text: 'Déposez le fichier audio ici ou cliquez pour sélectionner',
    drop_hint: 'Prend en charge mp3, wav, m4a, ogg, flac, mp4, mov, etc.',
    auto_detect: 'Détection automatique de la langue',
    fmt_verbose: 'Détaillé (verbose_json)',
    fmt_json: 'Texte brut (json)',
    fmt_text: 'Texte (text)',
    fmt_srt: 'Sous-titres (srt)',
    btn_transcribe: 'Transcrire',
    correct_toggle: 'Correction LLM',
    raw_json: 'JSON brut',
    corrected_label: 'Texte corrigé',
    api_examples: '📋 Exemples API',
    footer_prefix: 'Point de terminaison :',
    selected: 'Sélectionné',
    transcribing: 'Transcription en cours...',
    wait: 'Transcription en cours, veuillez patienter...',
    no_speech: '(Aucune parole détectée)',
    audio_label: 'Audio',
  },
  ru: {
    drop_text: 'Перетащите аудиофайл сюда или нажмите для выбора',
    drop_hint: 'Поддерживает mp3, wav, m4a, ogg, flac, mp4, mov и др.',
    auto_detect: 'Автоопределение языка',
    fmt_verbose: 'Подробно (verbose_json)',
    fmt_json: 'Текст (json)',
    fmt_text: 'Текст (text)',
    fmt_srt: 'Субтитры (srt)',
    btn_transcribe: 'Расшифровать',
    correct_toggle: 'Коррекция LLM',
    raw_json: 'Сырой JSON',
    corrected_label: 'Исправленный текст',
    api_examples: '📋 Примеры API',
    footer_prefix: 'Эндпоинт:',
    selected: 'Выбрано',
    transcribing: 'Расшифровка...',
    wait: 'Идёт расшифровка, пожалуйста, подождите...',
    no_speech: '(Речь не обнаружена)',
    audio_label: 'Аудио',
  },
  ar: {
    drop_text: 'اسحب ملف الصوت هنا أو انقر للاختيار',
    drop_hint: 'يدعم mp3, wav, m4a, ogg, flac, mp4, mov وغيرها',
    auto_detect: 'اكتشاف تلقائي للغة',
    fmt_verbose: 'مفصل (verbose_json)',
    fmt_json: 'نص عادي (json)',
    fmt_text: 'نص (text)',
    fmt_srt: 'ترجمة (srt)',
    btn_transcribe: 'نسخ',
    correct_toggle: 'تصحيح LLM',
    raw_json: 'JSON الخام',
    corrected_label: 'النص المصحح',
    api_examples: '📋 أمثلة API',
    footer_prefix: 'نقطة النهاية:',
    selected: 'مُختار',
    transcribing: 'جاري النسخ...',
    wait: 'جاري النسخ، يرجى الانتظار...',
    no_speech: '(لم يتم اكتشاف أي كلام)',
    audio_label: 'صوت',
  }
};

function detectBrowserLang() {
  const nav = (navigator.language || 'en').split('-')[0];
  return I18N[nav] ? nav : 'en';
}

let currentLang = localStorage.getItem('whisper-lang') || detectBrowserLang();

function applyLang(lang) {
  currentLang = lang;
  localStorage.setItem('whisper-lang', lang);
  document.getElementById('lang-switcher').value = lang;

  // Update plain text elements
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (I18N[lang] && I18N[lang][key]) {
      el.textContent = I18N[lang][key];
    }
  });

  // Rebuild selects (option textContent update is unreliable across browsers)
  rebuildSelects();

  if (transcribeBtn && !transcribeBtn.disabled) {
    transcribeBtn.textContent = I18N[lang].btn_transcribe;
  }
}

// Select option templates keyed by select id
const SELECT_OPTIONS = {
  fmt: [
    { value: 'verbose_json', key: 'fmt_verbose' },
    { value: 'json',          key: 'fmt_json' },
    { value: 'text',          key: 'fmt_text' },
    { value: 'srt',           key: 'fmt_srt' },
  ],
  lang: [
    { value: '',  key: 'auto_detect' },
    { value: 'zh', label: '🇨🇳 中文' },
    { value: 'en', label: '🇬🇧 English' },
    { value: 'ja', label: '🇯🇵 日本語' },
    { value: 'ko', label: '🇰🇷 한국어' },
    { value: 'fr', label: '🇫🇷 Français' },
    { value: 'de', label: '🇩🇪 Deutsch' },
    { value: 'es', label: '🇪🇸 Español' },
    { value: 'ru', label: '🇷🇺 Русский' },
    { value: 'ar', label: '🇸🇦 العربية' },
  ]
};

function rebuildSelects() {
  for (const [selId, options] of Object.entries(SELECT_OPTIONS)) {
    const sel = document.getElementById(selId);
    if (!sel) continue;
    const current = sel.value;
    sel.innerHTML = '';
    for (const opt of options) {
      const el = document.createElement('option');
      el.value = opt.value;
      el.textContent = opt.label || I18N[currentLang][opt.key] || '';
      sel.appendChild(el);
    }
    sel.value = current;
  }
}

function switchLang(lang) { applyLang(lang); }

// ═══════════════════════════════════════════════════════════════
// API base replacement
// ═══════════════════════════════════════════════════════════════
const API_BASE = window.location.origin;
document.querySelectorAll('pre code').forEach(el => {
  el.textContent = el.textContent.replace(/\\${API_BASE}/g, API_BASE);
});

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('file-info');
const transcribeBtn = document.getElementById('transcribeBtn');
const resultDiv = document.getElementById('result');
let selectedFile = null;
let lastRawData = null;

// Health check
fetch('/health').then(r=>r.json()).then(d=>{
  document.getElementById('status').textContent = `model=${d.model} · ${d.device} ${d.compute_type}`;
}).catch(()=>{});

// Apply saved language preference
applyLang(currentLang);

// Drag & drop
dropzone.addEventListener('click',()=>fileInput.click());
dropzone.addEventListener('dragover',e=>{e.preventDefault();dropzone.classList.add('dragover')});
dropzone.addEventListener('dragleave',()=>dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop',e=>{e.preventDefault();dropzone.classList.remove('dragover');handleFile(e.dataTransfer.files[0])});
fileInput.addEventListener('change',()=>handleFile(fileInput.files[0]));

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  fileInfo.style.display = 'block';
  fileInfo.textContent = `${I18N[currentLang].selected}: ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
  transcribeBtn.disabled = false;
}

transcribeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  transcribeBtn.disabled = true;
  transcribeBtn.textContent = I18N[currentLang].transcribing;
  resultDiv.style.display = 'block';
  resultDiv.innerHTML = `<div class="loading">⏳ ${I18N[currentLang].wait}</div>`;

  const fd = new FormData();
  fd.append('file', selectedFile);
  const lang = document.getElementById('lang').value;
  if (lang) fd.append('language', lang);
  fd.append('response_format', document.getElementById('fmt').value);
  if (document.getElementById('correctCheck').checked) {
    fd.append('correct', 'true');
  }

  try {
    const start = Date.now();
    const r = await fetch('/v1/audio/transcriptions', { method:'POST', body:fd });
    const data = await r.json();
    const elapsed = ((Date.now()-start)/1000).toFixed(1);

    if (!r.ok) { resultDiv.innerHTML = `<div class="err">❌ ${data.detail||r.statusText}</div>`; return; }

    lastRawData = data;
    renderResult(data, elapsed);
  } catch(e) {
    resultDiv.innerHTML = `<div class="err">❌ ${e.message}</div>`;
  } finally {
    transcribeBtn.disabled = false;
    transcribeBtn.textContent = I18N[currentLang].btn_transcribe;
  }
});

function switchTab(lang) {
  ['shell','python','js','java'].forEach(id => {
    document.getElementById('code-'+id).style.display = id===lang?'block':'none';
  });
  document.querySelectorAll('.tab').forEach((btn,i) => {
    const names = ['shell','python','js','java'];
    const active = names[i]===lang;
    btn.style.background = active?'var(--accent)':'var(--surface)';
    btn.style.color = active?'#1e1e2e':'var(--text)';
    btn.style.fontWeight = active?'600':'400';
  });
}

function renderResult(data, elapsed) {
  let html = '';
  if (data.language) html += `<div class="lang-badge">${data.language}${data.language_probability ? ' · '+Math.round(data.language_probability*100)+'%' : ''}</div>`;
  if (data.correction === 'completed') html += '<span style="display:inline-block;background:var(--green);color:#1e1e2e;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-bottom:8px;">✓ LLM corrected</span>';
  if (data.correction && data.correction !== 'completed') html += `<span style="display:inline-block;background:var(--red);color:#1e1e2e;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-bottom:8px;">⚠ ${data.correction}</span>`;
  html += `<div class="text">${data.text||I18N[currentLang].no_speech}</div>`;
  html += `<div class="sub" style="font-size:12px;margin-bottom:10px">⏱ ${elapsed}s · ${I18N[currentLang].audio_label} ${data.duration?data.duration.toFixed(1)+'s':''}</div>`;

  if (data.corrected_segments) {
    const correctedText = data.corrected_segments.map(cs => cs.corrected || cs.original).filter(Boolean).join(' ');
    if (correctedText) {
      html += '<div style="margin-top:12px;padding:14px;border-left:3px solid var(--green);background:#1e2e1e;border-radius:8px;">';
      html += `<div style="font-size:13px;font-weight:600;color:var(--green);margin-bottom:6px;">✓ ${I18N[currentLang].corrected_label}</div>`;
      html += `<div style="font-size:16px;line-height:1.7;white-space:pre-wrap;">${correctedText}</div>`;
      html += '</div>';
    }
  }

  if (data.segments && data.segments.length) {
    html += '<div class="segments">';
    for (let i = 0; i < data.segments.length; i++) {
      const seg = data.segments[i];
      const ts = seg.start.toFixed(1)+'s - '+seg.end.toFixed(1)+'s';
      let segHtml = `<div class="seg"><div class="ts">${ts}</div><div class="txt">${seg.text}</div></div>`;
      if (data.corrected_segments && data.corrected_segments[i] && data.corrected_segments[i].corrected) {
        segHtml += `<div class="seg" style="border-left:2px solid var(--green);padding-left:12px;color:var(--green);font-size:13px;">${data.corrected_segments[i].corrected}</div>`;
      }
      html += segHtml;
    }
    html += '</div>';
  }

  html += '<div style="margin-top:12px;text-align:right;">';
  html += `<button onclick="toggleRaw()" style="background:none;border:1px solid var(--border);color:var(--sub);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px;">📋 ${I18N[currentLang].raw_json}</button>`;
  html += '</div>';
  html += '<div id="rawJson" style="display:none;margin-top:8px;background:#11111b;border-radius:8px;padding:14px;overflow-x:auto;font-size:12px;line-height:1.5;max-height:500px;overflow-y:auto;"><pre style="margin:0;color:#cdd6f4;white-space:pre-wrap;word-break:break-all;">';
  html += escapeHtml(JSON.stringify(data, null, 2));
  html += '</pre></div>';

  resultDiv.innerHTML = html;
}

function toggleRaw() {
  const el = document.getElementById('rawJson');
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# LLM Correction
# ═══════════════════════════════════════════════════════════════════════════
async def llm_correct(text: str, language: str | None = None) -> str | None:
    """Send transcription text to LLM for correction. Returns corrected text or None.
    Disabled entirely unless LLM_MODEL is set in environment."""
    if not LLM_MODEL:
        return None

    lang_hint = f" in {language}" if language else ""
    prompt = (
        f"Correct the following speech-to-text transcription{lang_hint}. "
        f"Fix transcription errors, grammar, and punctuation. "
        f"Preserve the original meaning and style. "
        f"Output only the corrected text, nothing else:\n\n{text}"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
            )
            r.raise_for_status()
            corrected = r.json()["choices"][0]["message"]["content"].strip()
            log.info(f"LLM correction done ({LLM_MODEL}) — text reduced by {max(0, len(text)-len(corrected))} chars")
            return corrected
    except Exception as e:
        log.warning(f"LLM correction failed: {e}")
        return None


# ── Health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "model_load_time_s": round(model_load_time, 1),
    }


# ── Transcribe ───────────────────────────────────────────────────────────
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    correct: Optional[bool] = Form(False),
):
    """
    OpenAI-compatible transcription endpoint.

    - `file`: audio file (mp3, wav, m4a, ogg, flac, etc.)
    - `language`: language code (e.g. "en", "zh") or None for auto-detect
    - `response_format`: "json" (default), "text", "srt", "verbose_json"
    - `correct`: enable LLM correction (requires LLM_API_KEY env)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file extension
    allowed = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus",
               ".aac", ".wma", ".webm", ".mp4", ".mov", ".mkv"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext and ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    # Save uploaded file to temp
    suffix = ext or ".tmp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        log.info(f"Transcribing '{file.filename}' ({os.path.getsize(tmp_path)} bytes)"
                 f"{' lang=' + language if language else ''}")

        start = time.time()
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
        )

        # Consumer segments (generator -> list)
        seg_list = list(segments)
        full_text = " ".join(s.text.strip() for s in seg_list)
        elapsed = time.time() - start
        duration = info.duration

        log.info(f"Done in {elapsed:.1f}s ({duration/elapsed:.1f}x realtime)"
                 f" — lang={info.language}({info.language_probability:.2f})"
                 f" — {len(seg_list)} segments")

                # ── LLM Correction (per-segment) ───────────────────────────────
        corrected_segments: list[dict] | None = None
        correction = None
        if correct:
            if LLM_MODEL:
                corrected_segments = []
                for seg in seg_list:
                    seg_corrected = await llm_correct(seg.text.strip(), info.language)
                    if seg_corrected:
                        corrected_segments.append({
                            "id": seg.id,
                            "original": seg.text.strip(),
                            "corrected": seg_corrected,
                        })
                    else:
                        corrected_segments.append({
                            "id": seg.id,
                            "original": seg.text.strip(),
                            "corrected": None,
                        })
                if any(cs["corrected"] for cs in corrected_segments):
                    correction = "completed"
                else:
                    correction = "failed"
            else:
                correction = "skipped (LLM_MODEL not configured)"

        # ── Build response ────────────────────────────────────────────# ── Build response ────────────────────────────────────────────
        if response_format == "text":
            resp = {"text": full_text}
            if corrected_segments:
                resp["corrected_segments"] = corrected_segments
            if correction:
                resp["correction"] = correction
            return JSONResponse(resp)

        if response_format == "srt":
            srt_lines = []
            for i, seg in enumerate(seg_list, 1):
                srt_lines.append(str(i))
                srt_lines.append(f"{_fmt_srt(seg.start)} --> {_fmt_srt(seg.end)}")
                srt_lines.append(seg.text.strip())
                srt_lines.append("")
            resp = {"text": "\n".join(srt_lines)}
            if corrected_segments:
                resp["corrected_segments"] = corrected_segments
            if correction:
                resp["correction"] = correction
            return JSONResponse(resp)

        if response_format == "verbose_json":
            resp = {
                "text": full_text,
                "language": info.language,
                "duration": info.duration,
                "segments": [
                    {
                        "id": s.id,
                        "start": round(s.start, 2),
                        "end": round(s.end, 2),
                        "text": s.text.strip(),
                        "avg_logprob": round(s.avg_logprob, 4),
                        "no_speech_prob": round(s.no_speech_prob, 4),
                    }
                    for s in seg_list
                ],
            }
            if corrected_segments:
                resp["corrected_segments"] = corrected_segments
            if correction:
                resp["correction"] = correction
            return resp

        # Default: "json"
        resp = {"text": full_text}
        if corrected_segments:
            resp["corrected_segments"] = corrected_segments
        if correction:
            resp["correction"] = correction
        return resp

    except Exception as e:
        log.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


def _fmt_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ═══════════════════════════════════════════════════════════════════════════
# CLI Mode (offline transcription, no server)
# ═══════════════════════════════════════════════════════════════════════════
def transcribe_cli(
    audio_path: str,
    language: str | None = None,
    fmt: str = "text",
    model_size: str | None = None,
    output_path: str | None = None,
    correct: bool = False,
):
    """Transcribe a file directly to stdout (no HTTP server)."""
    m_size = model_size or MODEL_SIZE
    m_device = DEVICE
    m_compute = COMPUTE_TYPE
    if sys.platform == "darwin":
        m_device = "cpu"  # faster-whisper CPU-only on macOS; mlx-whisper has its own package
        m_compute = "int8"

    log.info(f"Loading model '{m_size}' (device={m_device}, compute={m_compute})...")
    start = time.time()
    model = WhisperModel(m_size, device=m_device, compute_type=m_compute)
    log.info(f"Model loaded in {time.time()-start:.1f}s")

    log.info(f"Transcribing: {audio_path}" + (f" [lang={language}]" if language else ""))
    start = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400),
    )
    seg_list = list(segments)
    full_text = " ".join(s.text.strip() for s in seg_list)
    elapsed = time.time() - start

    log.info(f"Done in {elapsed:.1f}s ({info.duration/elapsed:.1f}x realtime)"
             f" — {info.language}({info.language_probability:.2f}) — {len(seg_list)} segments")

    # ── LLM Correction (CLI) ────────────────────────────────────────
    corrected_segments = None
    correction = None
    if correct:
        if LLM_MODEL:
            log.info(f"LLM correction enabled, correcting {len(seg_list)} segments...")
            corrected_segments = []
            for seg in seg_list:
                try:
                    seg_corrected = asyncio.run(llm_correct(seg.text.strip(), info.language))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    seg_corrected = loop.run_until_complete(llm_correct(seg.text.strip(), info.language))
                    loop.close()
                corrected_segments.append({
                    "id": seg.id,
                    "original": seg.text.strip(),
                    "corrected": seg_corrected,
                })
            if any(cs["corrected"] for cs in corrected_segments):
                correction = "completed"
                log.info("LLM correction done ✅")
            else:
                correction = "failed"
        else:
            correction = "skipped (LLM_MODEL not configured)"

    if fmt == "srt":
        lines = []
        for i, seg in enumerate(seg_list, 1):
            lines.append(str(i))
            lines.append(f"{_fmt_srt(seg.start)} --> {_fmt_srt(seg.end)}")
            lines.append(seg.text.strip())
            lines.append("")
        output = "\n".join(lines)
    elif fmt == "json":
        resp = {"text": full_text}
        if corrected_segments:
            resp["corrected_segments"] = corrected_segments
        output = json.dumps(resp, ensure_ascii=False)
    elif fmt == "verbose_json":
        resp = {
            "text": full_text,
            "language": info.language,
            "duration": info.duration,
            "segments": [
                {"id": s.id, "start": round(s.start, 2), "end": round(s.end, 2),
                 "text": s.text.strip(), "avg_logprob": round(s.avg_logprob, 4)}
                for s in seg_list
            ],
        }
        if corrected_segments:
            resp["corrected_segments"] = corrected_segments
        output = json.dumps(resp, ensure_ascii=False, indent=2)
    else:  # text (default)
        output = full_text
        if corrected_segments:
            output += "\n\n--- Corrected ---\n" + str(corrected_segments)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        log.info(f"Saved to {output_path}")
    else:
        print(output)


# ── Main ─────────────────────────────────────────────────────────────────
def _parse_main_args():
    """Minimal argparse for CLI mode vs server mode."""
    p = argparse.ArgumentParser(description="Whisper ASR Service")
    p.add_argument("--transcribe", metavar="FILE", help="Transcribe a file and exit (CLI mode, no server)")
    p.add_argument("--language", default=None, help="Language code (e.g. en, zh, ja)")
    p.add_argument("--format", default="text", choices=["text", "json", "srt", "verbose_json"], help="Output format (default: text)")
    p.add_argument("--model", default=None, help=f"Model size (default: {MODEL_SIZE})")
    p.add_argument("--output", default=None, help="Save output to file instead of stdout")
    p.add_argument("--correct", action="store_true", help="Enable LLM correction (requires LLM_API_KEY)")
    p.add_argument("--host", default=HOST, help=f"Server bind address (default: {HOST})")
    p.add_argument("--port", type=int, default=PORT, help=f"Server port (default: {PORT})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_main_args()

    if args.transcribe:
        # CLI mode: transcribe one file and exit
        transcribe_cli(
            audio_path=args.transcribe,
            language=args.language,
            fmt=args.format,
            model_size=args.model,
            output_path=args.output,
            correct=args.correct,
        )
    else:
        # Server mode
        log.info(f"Starting Whisper ASR server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
