# Voice Memo Analyzer — CSC 391/691

A web application that records or accepts audio input, transcribes it using Azure AI Speech, analyzes it with Azure AI Language, and returns a spoken summary via Text-to-Speech.

## Architecture
Audio Input → Speech-to-Text (Azure AI Speech F0) → Text Analysis (Azure AI Language F0) → Text-to-Speech (Azure Neural TTS F0) → Results UI + Telemetry (App Insights F0)

## Setup Instructions

### 1. Clone the repo
```bash
git clone git@github.com:caitlinsofia11/csc391-speech.git
cd csc391-speech
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Fill in your Azure keys in .env
```

### 4. Provision Azure Resources
```bash
az group create --name csc391-speech-rg --location eastus
az cognitiveservices account create --name csc391-speech --resource-group csc391-speech-rg --kind SpeechServices --sku F0 --location eastus --yes
az cognitiveservices account create --name csc391-language --resource-group csc391-speech-rg --kind TextAnalytics --sku F0 --location eastus --yes
az monitor log-analytics workspace create --resource-group csc391-speech-rg --workspace-name csc391-logs --location eastus
az monitor app-insights component create --app csc391-insights --location eastus --resource-group csc391-speech-rg --workspace <WORKSPACE_ID>
```

### 5. Run locally
```bash
flask run --port 5001
```

### 6. Deploy to Azure
```bash
az webapp up --resource-group csc391-speech-rg --name csc391-speech-co22 --runtime "PYTHON:3.11" --sku F1
```

## Endpoints
- `POST /transcribe` — Upload audio, get transcript
- `POST /analyze` — Analyze text for entities, sentiment, key phrases
- `POST /process` — Full pipeline: audio in, results + TTS out
- `GET /telemetry-summary` — Session telemetry stats

## Live URL
https://csc391-speech-co22.azurewebsites.net