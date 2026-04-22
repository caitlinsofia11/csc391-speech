from dotenv import load_dotenv
load_dotenv()

from telemetry import init_telemetry
init_telemetry()

from flask import Flask, request, jsonify
import os
import time
import threading
import tempfile
import base64
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from opentelemetry import trace, metrics

app = Flask(__name__, static_folder='static')

# --- Clients ---
speech_key = os.environ.get("AZURE_SPEECH_KEY")
speech_region = os.environ.get("AZURE_SPEECH_REGION")
language_key = os.environ.get("AZURE_LANGUAGE_KEY")
language_endpoint = os.environ.get("AZURE_LANGUAGE_ENDPOINT")

text_analytics_client = TextAnalyticsClient(
    endpoint=language_endpoint,
    credential=AzureKeyCredential(language_key)
)

# --- Telemetry ---
tracer = trace.get_tracer("memo-analyzer")
meter = metrics.get_meter("memo-analyzer")

stt_confidence_gauge = meter.create_gauge("stt_confidence")
stt_duration_gauge = meter.create_gauge("stt_duration_seconds")
stt_word_count_gauge = meter.create_gauge("stt_word_count")
entity_count_gauge = meter.create_gauge("language_entity_count")
keyphrase_count_gauge = meter.create_gauge("language_keyphrase_count")
sentiment_gauge = meter.create_gauge("language_sentiment")
tts_char_count_gauge = meter.create_gauge("tts_char_count")
stage_stt_hist = meter.create_histogram("stage_stt_ms")
stage_language_hist = meter.create_histogram("stage_language_ms")
stage_tts_hist = meter.create_histogram("stage_tts_ms")

session_log = []

# --- Transcribe ---
def transcribe(audio_path):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.request_word_level_timestamps()
if audio_path.endswith('.webm') or audio_path.endswith('.ogg'):
    stream_format = speechsdk.audio.AudioStreamFormat.get_compressed_format_from_stream_type(
        speechsdk.AudioStreamContainerFormat.OGG_OPUS)
    stream = speechsdk.audio.PushAudioInputStream(stream_format)
    with open(audio_path, 'rb') as f:
        stream.write(f.read())
    stream.close()
    audio_config = speechsdk.AudioConfig(stream=stream)
else:
    audio_config = speechsdk.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = threading.Event()
    result_holder = {}
    all_results = []

    def recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            all_results.append(evt.result)

    def session_stopped(evt):
        done.set()

    def canceled(evt):
        result_holder['error'] = str(evt)
        done.set()

    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(session_stopped)
    recognizer.canceled.connect(canceled)
    recognizer.start_continuous_recognition()
    done.wait(timeout=60)
    recognizer.stop_continuous_recognition()

    if 'error' in result_holder and not all_results:
        raise Exception(f"Transcription canceled: {result_holder['error']}")

    if not all_results:
        raise Exception("No transcription result")

    import json
    full_transcript = " ".join([r.text for r in all_results])
    words = []
    total_confidence = 0.0

    for r in all_results:
        if r.json:
            j = json.loads(r.json)
            best = j.get('NBest', [{}])[0]
            total_confidence += best.get('Confidence', 0.0)
            words += [{"word": w.get("Word"), "offset": w.get("Offset", 0)/10000000,
                      "duration": w.get("Duration", 0)/10000000,
                      "confidence": w.get("Confidence", 0.0)}
                     for w in best.get("Words", [])]

    confidence = total_confidence / len(all_results) if all_results else 0.0
    duration = sum(r.duration.total_seconds() if hasattr(r.duration, 'total_seconds') else r.duration / 10000000 for r in all_results if r.duration)

    return {
        "transcript": full_transcript,
        "language": "en-US",
        "duration_seconds": duration,
        "confidence": confidence,
        "words": words
    }

# --- Analyze ---
def analyze(text):
    docs = [{"id": "1", "language": "en", "text": text}]

    kp_response = text_analytics_client.extract_key_phrases(docs)
    ner_response = text_analytics_client.recognize_entities(docs)
    sentiment_response = text_analytics_client.analyze_sentiment(docs)
    linked_response = text_analytics_client.recognize_linked_entities(docs)

    key_phrases = kp_response[0].key_phrases if not kp_response[0].is_error else []
    entities = [{"text": e.text, "category": e.category, "confidence": e.confidence_score}
                for e in (ner_response[0].entities if not ner_response[0].is_error else [])]
    sentiment_doc = sentiment_response[0] if not sentiment_response[0].is_error else None
    sentiment = {
        "label": sentiment_doc.sentiment if sentiment_doc else "neutral",
        "scores": {
            "positive": sentiment_doc.confidence_scores.positive if sentiment_doc else 0,
            "neutral": sentiment_doc.confidence_scores.neutral if sentiment_doc else 0,
            "negative": sentiment_doc.confidence_scores.negative if sentiment_doc else 0,
        }
    }
    linked = [{"name": e.name, "url": e.url}
              for e in (linked_response[0].entities if not linked_response[0].is_error else [])]

    return {
        "key_phrases": key_phrases,
        "entities": entities,
        "sentiment": sentiment,
        "linked_entities": linked
    }

# --- Synthesize ---
def synthesize(text):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data = result.audio_data
        return {
            "audio_base64": base64.b64encode(audio_data).decode("utf-8"),
            "char_count": len(text)
        }
    else:
        raise Exception(f"TTS failed: {result.reason}")

def build_summary(lang_result):
    kp = lang_result["key_phrases"]
    entities = lang_result["entities"]
    sentiment = lang_result["sentiment"]["label"]
    kp_str = ", ".join(kp[:3]) if kp else "none"
    entity_types = {}
    for e in entities:
        entity_types[e["category"]] = entity_types.get(e["category"], 0) + 1
    entity_summary = ", ".join([f"{v} {k}" for k, v in entity_types.items()])
    if entity_summary:
        return (f"Your memo mentions {len(kp)} key topics: {kp_str}. "
                f"The overall tone is {sentiment}. "
                f"I detected {entity_summary}.")
    return (f"Your memo mentions {len(kp)} key topics: {kp_str}. "
            f"The overall tone is {sentiment}.")

# --- Routes ---
@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files["audio"]
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        try:
            result = transcribe(tmp.name)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(tmp.name)
    return jsonify(result)

@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = analyze(data["text"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)

@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files["audio"]
    audio_format = file.filename.rsplit(".", 1)[-1].lower()
    suffix = "." + audio_format

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        audio_path = tmp.name

    try:
        with tracer.start_as_current_span("pipeline.process") as root_span:
            root_span.set_attribute("audio.format", audio_format)

            with tracer.start_as_current_span("stage.speech_to_text") as stt_span:
                t0 = time.perf_counter()
                stt_result = transcribe(audio_path)
                stt_ms = (time.perf_counter() - t0) * 1000
                stt_span.set_attribute("stt.confidence", stt_result["confidence"])
                stt_span.set_attribute("duration_ms", stt_ms)

            with tracer.start_as_current_span("stage.language_analysis") as lang_span:
                t0 = time.perf_counter()
                lang_result = analyze(stt_result["transcript"])
                lang_ms = (time.perf_counter() - t0) * 1000
                lang_span.set_attribute("entity_count", len(lang_result["entities"]))
                lang_span.set_attribute("duration_ms", lang_ms)

            with tracer.start_as_current_span("stage.text_to_speech") as tts_span:
                t0 = time.perf_counter()
                summary = build_summary(lang_result)
                tts_result = synthesize(summary)
                tts_ms = (time.perf_counter() - t0) * 1000
                tts_span.set_attribute("char_count", len(summary))
                tts_span.set_attribute("duration_ms", tts_ms)

            attrs = {"audio_format": audio_format, "language": stt_result["language"]}
            stt_confidence_gauge.set(stt_result["confidence"], attrs)
            stt_duration_gauge.set(stt_result["duration_seconds"], attrs)
            stt_word_count_gauge.set(len(stt_result["transcript"].split()), attrs)
            entity_count_gauge.set(len(lang_result["entities"]), attrs)
            keyphrase_count_gauge.set(len(lang_result["key_phrases"]), attrs)
            sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            sentiment_gauge.set(sentiment_map.get(lang_result["sentiment"]["label"], 0.0), attrs)
            tts_char_count_gauge.set(tts_result["char_count"], attrs)
            stage_stt_hist.record(stt_ms, attrs)
            stage_language_hist.record(lang_ms, attrs)
            stage_tts_hist.record(tts_ms, attrs)

            session_log.append({
                "confidence": stt_result["confidence"],
                "language": stt_result["language"],
                "entity_count": len(lang_result["entities"]),
                "sentiment": lang_result["sentiment"]["label"],
                "stt_ms": stt_ms,
                "language_ms": lang_ms,
                "tts_ms": tts_ms,
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(audio_path)

    return jsonify({**stt_result, **lang_result, **tts_result, "summary": summary})

@app.route("/telemetry-summary", methods=["GET"])
def telemetry_summary():
    if not session_log:
        return jsonify({"message": "No calls yet"})
    import statistics
    confidences = [e["confidence"] for e in session_log]
    return jsonify({
        "total_calls": len(session_log),
        "avg_confidence": round(statistics.mean(confidences), 3),
        "min_confidence": round(min(confidences), 3),
        "p95_stt_ms": round(sorted([e["stt_ms"] for e in session_log])[int(len(session_log)*0.95)-1], 1),
        "sentiment_breakdown": {s: sum(1 for e in session_log if e["sentiment"]==s)
                                for s in ["positive","neutral","negative"]},
        "calls": session_log[-10:]
    })

@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
