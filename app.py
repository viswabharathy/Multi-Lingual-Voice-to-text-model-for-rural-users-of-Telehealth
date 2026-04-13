# ============================================================
# 🔒 ENV FIXES (MUST BE FIRST)
# ============================================================
import os
os.environ["TRANSFORMERS_NO_ONNX"] = "1"
os.environ["ORT_DISABLE_ARENA"] = "1"

# ============================================================
# 🔹 IMPORTS
# ============================================================
import tempfile
import shutil
import zipfile
import urllib.request

import torch
import librosa
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from transformers import (
    WhisperProcessor,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from IndicTransToolkit.processor import IndicProcessor
from pydub import AudioSegment

# ============================================================
# 🔹 BASE DIR
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 🔹 AUTO DOWNLOAD FFMPEG (WINDOWS SAFE)
# ============================================================
FFMPEG_DIR = os.path.join(BASE_DIR, "ffmpeg")
FFMPEG_BIN = os.path.join(FFMPEG_DIR, "bin", "ffmpeg.exe")
FFPROBE_BIN = os.path.join(FFMPEG_DIR, "bin", "ffprobe.exe")

def ensure_ffmpeg():
    if os.path.exists(FFMPEG_BIN) and os.path.exists(FFPROBE_BIN):
        return True

    print("⬇️ Downloading FFmpeg locally...")
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = os.path.join(BASE_DIR, "ffmpeg.zip")

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(BASE_DIR)

    extracted = next(
        d for d in os.listdir(BASE_DIR)
        if d.startswith("ffmpeg") and "essentials" in d
    )

    shutil.move(os.path.join(BASE_DIR, extracted), FFMPEG_DIR)
    os.remove(zip_path)
    return True

FFMPEG_AVAILABLE = ensure_ffmpeg()
os.environ["PATH"] = os.path.join(FFMPEG_DIR, "bin") + os.pathsep + os.environ["PATH"]

# ============================================================
# 🔹 PATHS
# ============================================================
WHISPER_ONNX_PATH = os.path.join(BASE_DIR, "whisper_kannada_onnx")
INDIC_CONFORMER_PATH = os.path.join(BASE_DIR, "indic_conformer")
INDICTRANS2_PATH = os.path.join(BASE_DIR, "indictrans2")

# ============================================================
# 🔹 DEVICE CONFIG
# ============================================================
ASR_DEVICE = "cpu"
MT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_WHISPER_GPU = torch.cuda.is_available()

# ============================================================
# 🔹 FASTAPI APP
# ============================================================
app = FastAPI(
    title="Telehealth ASR Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🔹 LOAD INDIC TRANS 2 (SAFE)
# ============================================================
print("🔄 Loading IndicTrans2...")

mt_tokenizer = AutoTokenizer.from_pretrained(
    INDICTRANS2_PATH,
    local_files_only=True,
    trust_remote_code=True
)

mt_model = AutoModelForSeq2SeqLM.from_pretrained(
    INDICTRANS2_PATH,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16 if MT_DEVICE == "cuda" else torch.float32,
    device_map="auto" if MT_DEVICE == "cuda" else None
).eval()

mt_processor = IndicProcessor(inference=True)

print("✅ IndicTrans2 loaded")

# ============================================================
# 🔹 LOAD INDIC CONFORMER
# ============================================================
print("🔄 Loading IndicConformer...")

asr_config = AutoConfig.from_pretrained(
    INDIC_CONFORMER_PATH,
    trust_remote_code=True,
    local_files_only=True
)

asr_model = AutoModel.from_pretrained(
    INDIC_CONFORMER_PATH,
    config=asr_config,
    trust_remote_code=True,
    local_files_only=True
).to(ASR_DEVICE).eval()

print("✅ IndicConformer loaded")

# ============================================================
# 🔹 LOAD WHISPER ONNX (KANNADA)
# ============================================================
print("🔄 Loading Whisper Kannada...")

PROVIDERS = ["CPUExecutionProvider"]
if USE_WHISPER_GPU:
    PROVIDERS.insert(0, "CUDAExecutionProvider")

whisper_processor = WhisperProcessor.from_pretrained(
    WHISPER_ONNX_PATH,
    local_files_only=True
)

whisper_encoder = ort.InferenceSession(
    os.path.join(WHISPER_ONNX_PATH, "encoder_model.onnx"),
    providers=PROVIDERS
)

whisper_decoder = ort.InferenceSession(
    os.path.join(WHISPER_ONNX_PATH, "decoder_model.onnx"),
    providers=PROVIDERS
)

print("✅ Whisper Kannada loaded")

# ============================================================
# 🔹 LANGUAGE MAP
# ============================================================
LANG_MAP = {
    "kannada": ("kn", "kan_Knda"),
    "hindi": ("hi", "hin_Deva"),
    "telugu": ("te", "tel_Telu"),
    "tamil": ("ta", "tam_Taml"),
    "malayalam": ("ml", "mal_Mlym"),
}

# ============================================================
# 🔹 UTILS
# ============================================================
def convert_to_wav(path):
    wav = tempfile.mktemp(suffix=".wav")
    audio = AudioSegment.from_file(path)
    audio.set_frame_rate(16000).set_channels(1).export(wav, format="wav")
    return wav

def load_audio(wav):
    audio, _ = librosa.load(wav, sr=16000, duration=15)
    return librosa.effects.trim(audio, top_db=20)[0]

# ============================================================
# 🔹 ASR FUNCTIONS
# ============================================================
def run_indic(audio, lang):
    wav = torch.tensor(audio).unsqueeze(0).to(ASR_DEVICE)
    with torch.no_grad():
        return asr_model(wav, LANG_MAP[lang][0], "ctc").strip()

def run_whisper(audio):
    inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="np")
    enc = whisper_encoder.run(None, {"input_features": inputs.input_features})[0]

    ids = np.array([[whisper_processor.tokenizer.bos_token_id]], dtype=np.int64)
    tokens = []

    for _ in range(96):
        logits = whisper_decoder.run(
            None,
            {"input_ids": ids, "encoder_hidden_states": enc}
        )[0]

        next_id = int(np.argmax(logits[:, -1, :]))
        if next_id == whisper_processor.tokenizer.eos_token_id:
            break

        tokens.append(next_id)
        ids = np.concatenate([ids, [[next_id]]], axis=1)

    return whisper_processor.tokenizer.decode(tokens, skip_special_tokens=True)

# ============================================================
# 🔹 TRANSLATION (SAFE)
# ============================================================
def translate_to_english(text, lang):
    if not text.strip():
        return ""

    _, src_lang = LANG_MAP[lang]

    batch = mt_processor.preprocess_batch(
        [text],
        src_lang=src_lang,
        tgt_lang="eng_Latn"
    )

    inputs = mt_tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(MT_DEVICE)

    with torch.no_grad():
        outputs = mt_model.generate(
            **inputs,
            max_length=256,
            num_beams=5
        )

    decoded = mt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return mt_processor.postprocess_batch(decoded, lang="eng_Latn")[0]

# ============================================================
# 🔹 HEALTH
# ============================================================
@app.get("/")
def health():
    return {
        "status": "running",
        "engines": ["indic", "whisper"],
        "languages": list(LANG_MAP.keys()),
        "ffmpeg": FFMPEG_AVAILABLE
    }

# ============================================================
# 🔹 MAIN API
# ============================================================
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    engine: str = Query("indic"),
    language: str = Query("kannada"),
    translate: bool = Query(False)
):
    if language not in LANG_MAP:
        raise HTTPException(400, "Unsupported language")

    suffix = os.path.splitext(file.filename)[-1].lower()
    input_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            input_path = tmp.name

        wav_path = convert_to_wav(input_path) if suffix != ".wav" else input_path
        audio = load_audio(wav_path)

        if engine == "whisper":
            if language != "kannada":
                raise HTTPException(400, "Whisper supports Kannada only")
            text = run_whisper(audio)
        else:
            text = run_indic(audio, language)

        result = {
            "engine": engine,
            "language": language,
            "transcription": text
        }

        if translate:
            result["english_translation"] = translate_to_english(text, language)

        return result

    finally:
        for p in [input_path, wav_path]:
            if p and os.path.exists(p):
                os.remove(p)
