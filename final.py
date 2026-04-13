# ============================================================
# TELEHEALTH BACKEND SERVER (app.py)
# ============================================================

import os
import gdown
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import librosa
import numpy as np
import onnxruntime as ort
from transformers import (
    WhisperProcessor,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from IndicTransToolkit.processor import IndicProcessor
import tempfile
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Your Google Drive Folder IDs (from your links)
DRIVE_FOLDERS = {
    'whisper': {
        'id': '1I1cuDRpZiCnzyyqpHEjMN9R91tFZ06iZ',
        'local_path': os.path.join(BASE_DIR, 'whisper_kannada_onnx')
    },
    'indictrans': {
        'id': '1l9W1vMPVjYYSC4BUbp_n187MKbRjewGq',
        'local_path': os.path.join(BASE_DIR, 'indictrans2')
    },
    'conformer': {
        'id': '1zKiQ7oxBP2XKt1P2c9FCXyIsWCF65Rxr',
        'local_path': os.path.join(BASE_DIR, 'indic_conformer')
    }
}

# Language Map
LANG_MAP = {
    "kannada": ("kn", "kan_Knda"),
    "hindi": ("hi", "hin_Deva"),
    "telugu": ("te", "tel_Telu"),
    "tamil": ("ta", "tam_Taml"),
    "malayalam": ("ml", "mal_Mlym"),
}

# ============================================================
# DOWNLOAD MODELS FROM GOOGLE DRIVE (FIRST TIME ONLY)
# ============================================================

def download_models():
    """Download all models from Google Drive if not present"""
    print("🔍 Checking for models...")
    
    all_exist = True
    for model_name, config in DRIVE_FOLDERS.items():
        if os.path.exists(config['local_path']) and os.listdir(config['local_path']):
            print(f"✅ {model_name} exists locally")
        else:
            print(f"❌ {model_name} not found, will download")
            all_exist = False
    
    if all_exist:
        print("✅ All models present!")
        return
    
    print("\n🚀 Downloading models from Google Drive (7GB total)...")
    print("⚠️ This will happen ONLY ONCE on first startup\n")
    
    for model_name, config in DRIVE_FOLDERS.items():
        if os.path.exists(config['local_path']) and os.listdir(config['local_path']):
            continue
            
        print(f"📥 Downloading {model_name}...")
        os.makedirs(config['local_path'], exist_ok=True)
        
        url = f"https://drive.google.com/drive/folders/{config['id']}"
        try:
            gdown.download_folder(
                url,
                output=config['local_path'],
                quiet=False,
                use_cookies=False
            )
            print(f"✅ {model_name} downloaded")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            raise
    
    print("\n✅ All models downloaded successfully!")

# Download models on startup
download_models()

# ============================================================
# INITIALIZE FASTAPI
# ============================================================

app = FastAPI(title="Telehealth ASR API", version="1.0")

# Enable CORS for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your Android app
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODELS (HAPPENS ONCE AT SERVER START)
# ============================================================

print("\n📦 Loading models...")

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load IndicTrans2
print("Loading IndicTrans2...")
mt_tokenizer = AutoTokenizer.from_pretrained(
    DRIVE_FOLDERS['indictrans']['local_path'],
    local_files_only=True,
    trust_remote_code=True
)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(
    DRIVE_FOLDERS['indictrans']['local_path'],
    local_files_only=True,
    trust_remote_code=True
).to(device).eval()
mt_processor = IndicProcessor(inference=True)
print("✅ IndicTrans2 loaded")

# Load IndicConformer
print("Loading IndicConformer...")
asr_config = AutoConfig.from_pretrained(
    DRIVE_FOLDERS['conformer']['local_path'],
    trust_remote_code=True,
    local_files_only=True
)
asr_model = AutoModel.from_pretrained(
    DRIVE_FOLDERS['conformer']['local_path'],
    config=asr_config,
    trust_remote_code=True,
    local_files_only=True
).to("cpu").eval()
print("✅ IndicConformer loaded")

# Load Whisper
print("Loading Whisper Kannada...")
whisper_processor = WhisperProcessor.from_pretrained(
    DRIVE_FOLDERS['whisper']['local_path'],
    local_files_only=True
)
whisper_encoder = ort.InferenceSession(
    os.path.join(DRIVE_FOLDERS['whisper']['local_path'], "encoder_model.onnx"),
    providers=["CPUExecutionProvider"]
)
whisper_decoder = ort.InferenceSession(
    os.path.join(DRIVE_FOLDERS['whisper']['local_path'], "decoder_model.onnx"),
    providers=["CPUExecutionProvider"]
)
print("✅ Whisper loaded")
print("🚀 All models ready!\n")

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def health_check():
    """Check if server is running"""
    return {
        "status": "online",
        "models": list(DRIVE_FOLDERS.keys()),
        "languages": list(LANG_MAP.keys())
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query("kannada", enum=list(LANG_MAP.keys())),
    engine: str = Query("indic", enum=["indic", "whisper"])
):
    """
    Transcribe audio file and translate to English
    """
    if language not in LANG_MAP:
        raise HTTPException(400, "Unsupported language")
    
    if engine == "whisper" and language != "kannada":
        raise HTTPException(400, "Whisper supports Kannada only")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        input_path = tmp.name
    
    try:
        # Load and process audio
        audio, _ = librosa.load(input_path, sr=16000)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Run ASR
        if engine == "whisper":
            text = run_whisper(audio)
        else:
            text = run_indic_conformer(audio, language)
        
        # Translate to English
        english = translate_to_english(text, language)
        
        return {
            "success": True,
            "transcription": text,
            "translation": english,
            "language": language,
            "engine": engine
        }
        
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        os.unlink(input_path)

@app.post("/transcribe/file")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Query("kannada", enum=list(LANG_MAP.keys()))
):
    """
    Alternative endpoint that returns both transcription and translation
    """
    result = await transcribe(file, language, "indic")
    return result

@app.get("/languages")
def get_languages():
    """Get supported languages"""
    return {
        "languages": [
            {
                "name": lang,
                "code": LANG_MAP[lang][0],
                "script": LANG_MAP[lang][1]
            }
            for lang in LANG_MAP.keys()
        ]
    }

# ============================================================
# MODEL FUNCTIONS
# ============================================================

def run_indic_conformer(audio, language):
    """Run IndicConformer ASR"""
    wav = torch.tensor(audio).unsqueeze(0)
    with torch.no_grad():
        text = asr_model(wav, LANG_MAP[language][0], "ctc")
    return text.strip()

def run_whisper(audio):
    """Run Whisper ASR"""
    inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="np")
    enc = whisper_encoder.run(None, {"input_features": inputs.input_features})[0]
    
    ids = np.array([[whisper_processor.tokenizer.bos_token_id]])
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

def translate_to_english(text, language):
    """Translate text to English"""
    if not text.strip():
        return ""
    
    _, src_lang = LANG_MAP[language]
    
    processed = mt_processor.preprocess_batch(
        [text],
        src_lang=src_lang,
        tgt_lang="eng_Latn"
    )
    
    inputs = mt_tokenizer(
        processed,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(mt_model.device)
    
    with torch.no_grad():
        outputs = mt_model.generate(
            **inputs,
            max_length=256,
            num_beams=5
        )
    
    decoded = mt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return mt_processor.postprocess_batch(decoded, lang="eng_Latn")[0]

# ============================================================
# RUN SERVER (Hugging Face Spaces compatible)
# ============================================================

import gradio as gr
from fastapi import FastAPI
import uvicorn

# create a simple UI so the Space runs correctly
def info():
    return "Telehealth ASR backend is running"

demo = gr.Interface(
    fn=info,
    inputs=None,
    outputs="text",
    title="Telehealth Backend API",
    description="Backend for multilingual telehealth speech recognition"
)

# mount FastAPI inside Gradio app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)