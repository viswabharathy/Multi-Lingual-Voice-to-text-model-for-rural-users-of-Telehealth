# 🏥 Multi-Lingual Voice-to-Text System for Rural Telehealth

---

## 📌 Overview

This project is a **Multilingual Speech-to-Text and Translation System** designed for rural telehealth applications.

---

## 🎯 Key Features

* 🎤 Speech → Text
* 🌍 Translation to English
* 📱 Android + Backend Integration
* ⚡ ONNX Optimized Models

---

## 🌐 Supported Languages

* Kannada
* Hindi
* Telugu
* Tamil
* Malayalam

---

## 🧠 System Architecture

Android App → FastAPI Backend → Hugging Face Models → Output

---

## 🔧 Components

### 📱 Android App

* Records or uploads audio
* Sends data to backend

### ⚙️ Backend (FastAPI)

* Handles API requests
* Runs ML models
* Returns output

### 🤖 Models

* IndicConformer → Speech-to-text
* IndicTrans2 → Translation
* Whisper ONNX → Kannada

---

## 🤖 Models

* Translation: https://huggingface.co/vragipin/translation-model
* Transcription: https://huggingface.co/vragipin/transcription-model
* Whisper Kannada: https://huggingface.co/vragipin/whisper-kannada-onnx-model

⚠️ First run downloads ~13GB

---

# 💻 Backend Setup (Step-by-Step)

---

## 1️⃣ Install Python (IMPORTANT)

👉 Install **Python 3.10 ONLY**

Download from:
https://www.python.org/downloads/release/python-3100/

✔ During installation:

* ✅ Check **Add Python to PATH**

---

## 2️⃣ Clone Repository

```
git clone https://github.com/vragipin/Multi-Lingual-Voice-to-text-model-for-rural-users-of-Telehealth
cd Multi-Lingual-Voice-to-text-model-for-rural-users-of-Telehealth/telehealth_backend
```

---

## 3️⃣ Create Virtual Environment

```
python -m venv venv310
```

---

## 4️⃣ Install Visual Studio C++ Build Tools

👉 Required to compile dependencies (VERY IMPORTANT)

Install Visual C++ Build Tools (Try this first)

Download from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### During installation, select:

* ✅ Desktop development with C++ workload

### Under Individual components:

* ✅ Windows 10/11 SDK
* ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools

### Final Step:

* 🔄 Restart your computer

---

## 5️⃣ Activate Environment & Install Requirements

```
venv310\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6️⃣ Run Backend

```
python main.py
```

✔ Output:

```
Uvicorn running on http://0.0.0.0:8000
```

---

## 7️⃣ Access Backend

* http://127.0.0.1:8000
* http://127.0.0.1:8000/docs

---

# 🌍 Public Access using Ngrok

---

## 8️⃣ Install Ngrok using Winget

Open Command Prompt (Admin):

```
winget install ngrok.ngrok
```

---

## 9️⃣ Authenticate Ngrok

Go to: https://dashboard.ngrok.com/get-started/your-authtoken

Copy your token and run:

```
ngrok config add-authtoken YOUR_AUTHTOKEN
```

---

## 🔟 Start Ngrok

```
ngrok http 8000
```

---

## 🌐 Get Public URL

Example output:

```
Forwarding: https://xxxxx.ngrok-free.dev -> http://localhost:8000
```

---

## 1️⃣1️⃣ Use URL in Application

👉 Copy this URL and paste in your Android app backend/base URL

Example:

```
BASE_URL = https://xxxxx.ngrok-free.dev
```

---

## 🔗 Test URLs

* API → https://xxxxx.ngrok-free.dev
* Swagger → https://xxxxx.ngrok-free.dev/docs

---

# 📱 Android App Usage

---

## 🔽 Install APK

```
app.apk
```

---

## 📲 Steps

1. Open app
2. Select language
3. Choose model:

   * Indic → All languages
   * Whisper → Kannada
4. Record / Upload audio
5. Click **Transcribe**

---

## ✅ Output

* 📝 Transcription
* 🌍 English Translation

---

# 🧪 Testing Workflow

1. Run backend
2. Wait for models loading
3. Start ngrok
4. Copy URL
5. Paste in Android app
6. Test

---

# ⚠️ Important Notes

* First run takes 10–20 minutes
* Models (~13GB) download once
* Works offline after first run
* Keep backend + ngrok running

---

# 🚀 Features

* Multilingual speech recognition
* Real-time translation
* FastAPI backend
* Android integration
* ONNX acceleration

---

# 👨‍💻 Authors

* Dr. Viswa Bharathy A. M
* Vishnuvardhan Reddy
* Bharath Kumar
* Varshith
* Masineni Kowshik

---
