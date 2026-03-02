import os, tempfile, base64, json
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import soundfile as sf
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from openai import OpenAI

# CONFIG
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_DIR = Path(__file__).parent.resolve()
app = Flask(__name__, template_folder="templates", static_folder="static")
sentiment = SentimentIntensityAnalyzer()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------------- Voice Classifier -----------------
def make_synthetic_XY(n=600, rs=0):
    rng = np.random.RandomState(rs)
    X, y = [], []
    for i in range(n):
        label = 1 if rng.rand() < 0.4 else 0
        if label == 0:
            rms_mean = rng.normal(0.02, 0.005); rms_std = abs(rng.normal(0.005,0.003))
            pitch_std = abs(rng.normal(8,3)); voiced = rng.uniform(0.3,0.7)
        else:
            rms_mean = rng.normal(0.03, 0.01); rms_std = abs(rng.normal(0.02,0.01))
            pitch_std = abs(rng.normal(22,8)); voiced = rng.uniform(0.4,0.9)
        mfcc = rng.normal(-32,4)
        X.append([rms_mean,rms_std,pitch_std,mfcc,voiced])
        y.append(label)
    return np.array(X), np.array(y)

X, y = make_synthetic_XY(800, rs=1)
MODEL = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=80, random_state=0))])
MODEL.fit(X, y)
DISTRESS_EN = [
  "mujhe bahut tension ho rahi hai",
  "mujhe darr lag raha hai",
  "main bohot pareshaan hoon",
  "meri tabiyat theek nahi lag rahi",
  "main bilkul theek nahi hoon",
  "mujhe samajh nahi aa raha kya karu",
  "maine control kho diya hai",
  "main toot sa raha hoon",
  "mera mann bahut ghabra raha hai",
  "mujhe ghabrahat ho rahi hai",
  "main handle nahi kar paa raha",
  "mujhe sab kuch heavy lag raha hai",
  "main mentally thak gaya hoon",
  "main bahut thak chuka hoon",
  "mera dimaag bandh ho gaya hai",
  "main akela mehsoos kar raha hoon",
  "mujhe kuch bhi sahi nahi lag raha",
  "main bahut udaas hoon",
  "mujhse kuch ho nahi raha",
  "main har gaya hoon",
  "main give up karna chahta hoon",
  "sab kuch mushkil lag raha hai",
  "jeena mushkil ho gaya hai",
  "mujhe bas rona aa raha hai",
  "meri himmat toot gayi hai",
  "main bahut pressure me hoon",
  "mujhe kisi se baat karni hai",
  "mujhe help chahiye",
  "main bahut insecure feel kar raha hoon",
  "main anxiety feel kar raha hoon",
  "meri heartbeat bahut fast hai",
  "main panic me hoon",
  "mujhe panic ho raha hai",
  "main bohot zyada stress me hoon",
  "mujhe lag raha hai kuch bura hone wala hai",
  "mujhse concentrate nahi ho raha",
  "main theek se so nahi pa raha",
  "mujhe khud par bharosa nahi raha",
  "mera dimaag bahut bhaari lag raha hai",
  "main stress me dub raha hoon",
  "mujhe lag raha hai main fail ho jaunga",
  "main emotionally down hoon",
  "main mentally stable nahi hoon",
  "mere andar dar baitha hua hai",
  "mujhe bilkul accha nahi lag raha",
  "main toot gaya hoon",
  "main bohot weak feel kar raha hoon",
  "main apne emotions control nahi kar paa raha",
  "main hil gaya hoon",
  "yeh sab mere bas ka nahi",
  "main is waqt bilkul stable nahi hoon"
]
DISTRESS_HI = [
  "mujhe bahut tension ho rahi hai",
  "mujhe darr lag raha hai",
  "main bohot pareshaan hoon",
  "meri tabiyat theek nahi lag rahi",
  "main bilkul theek nahi hoon",
  "mujhe samajh nahi aa raha kya karu",
  "maine control kho diya hai",
  "main toot sa raha hoon",
  "mera mann bahut ghabra raha hai",
  "mujhe ghabrahat ho rahi hai",
  "main handle nahi kar paa raha",
  "mujhe sab kuch heavy lag raha hai",
  "main mentally thak gaya hoon",
  "main bahut thak chuka hoon",
  "mera dimaag bandh ho gaya hai",
  "main akela mehsoos kar raha hoon",
  "mujhe kuch bhi sahi nahi lag raha",
  "main bahut udaas hoon",
  "mujhse kuch ho nahi raha",
  "main har gaya hoon",
  "main give up karna chahta hoon",
  "sab kuch mushkil lag raha hai",
  "jeena mushkil ho gaya hai",
  "mujhe bas rona aa raha hai",
  "meri himmat toot gayi hai",
  "main bahut pressure me hoon",
  "mujhe kisi se baat karni hai",
  "mujhe help chahiye",
  "main bahut insecure feel kar raha hoon",
  "main anxiety feel kar raha hoon",
  "meri heartbeat bahut fast hai",
  "main panic me hoon",
  "mujhe panic ho raha hai",
  "main bohot zyada stress me hoon",
  "mujhe lag raha hai kuch bura hone wala hai",
  "mujhse concentrate nahi ho raha",
  "main theek se so nahi pa raha",
  "mujhe khud par bharosa nahi raha",
  "mera dimaag bahut bhaari lag raha hai",
  "main stress me dub raha hoon",
  "mujhe lag raha hai main fail ho jaunga",
  "main emotionally down hoon",
  "main mentally stable nahi hoon",
  "mere andar dar baitha hua hai",
  "mujhe bilkul accha nahi lag raha",
  "main toot gaya hoon",
  "main bohot weak feel kar raha hoon",
  "main apne emotions control nahi kar paa raha",
  "main hil gaya hoon",
  "yeh sab mere bas ka nahi",
  "main is waqt bilkul stable nahi hoon"
]
# ---------------- Audio -----------------
def save_webm_to_wav(file_bytes: bytes):
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tmp_in.write(file_bytes); tmp_in.flush(); tmp_in.close()
    y, sr_audio = librosa.load(tmp_in.name, sr=16000)
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_wav.name, y, sr_audio, subtype='PCM_16')
    os.unlink(tmp_in.name)
    return tmp_wav.name

def extract_features(wav_path):
    y, sr_audio = librosa.load(wav_path, sr=16000)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean, rms_std = float(np.mean(rms)), float(np.std(rms))
    try:
        pitches = librosa.yin(y, fmin=75, fmax=400, sr=sr_audio)
        pitches = pitches[~np.isnan(pitches)]
        pitch_std = float(np.std(pitches)) if len(pitches)>0 else 0.0
    except:
        pitch_std = 0.0
    mfcc = librosa.feature.mfcc(y=y, sr=sr_audio, n_mfcc=13)
    mfcc_mean = float(np.mean(mfcc))
    voiced_prop = float(np.mean(rms > np.percentile(rms,25)))
    return np.array([rms_mean,rms_std,pitch_std,mfcc_mean,voiced_prop],dtype=float)

def analyze_text(text, lang='en'):
    t = (text or "").lower()
    phrases = DISTRESS_EN if lang=='en' else DISTRESS_HI
    found = [p for p in phrases if p in t]
    vader = sentiment.polarity_scores(t) if t else {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    text_score = int(min(100, vader["neg"]*100 + len(found)*18))
    return text_score, found

# ---------------- GAD-7 -----------------
def gad7_score(answers):
    total = int(sum(int(x) for x in answers))
    percent = int(round((total/21.0)*100))
    if total <=4:
        label={"en":"Minimal anxiety","hi":"कम चिंता"}
    elif total<=9:
        label={"en":"Mild anxiety","hi":"हल्की चिंता"}
    elif total<=14:
        label={"en":"Moderate anxiety","hi":"मध्यम चिंता"}
    else:
        label={"en":"Severe anxiety","hi":"गंभीर चिंता"}
    return total, percent, label

# ---------------- Spider Chart -----------------
def create_spider_chart(scores):
    categories = list(scores.keys())
    values = list(scores.values())
    N = len(categories)
    values += values[:1]
    angles = [n/float(N)*2*np.pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    ax.set_ylim(0, 100)
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#667eea')
    ax.fill(angles, values, '#667eea', alpha=0.4)
    ax.set_facecolor('#f0f8ff')
    ax.grid(True, color='#ccc', linestyle='--', linewidth=0.5)
    
    buf = tempfile.SpooledTemporaryFile()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_data

# ---------------- AI Report -----------------
def ai_analysis(transcript, detected, gad_total, gad_percent, voice_score, lang='en'):
    if not client:
        print("Warning: OpenAI API key not configured")
        return get_fallback_response(gad_total, gad_percent, voice_score, lang)
    
    prompt = f"""
You are a compassionate mental health assistant. Based on the user's assessment:

Transcript: {transcript}
Distress keywords detected: {detected}
GAD-7 score: {gad_total}/21 ({gad_percent}%)
Voice stress score: {voice_score}%
Language: {lang}

Generate a JSON response with:
1. "report_en": A warm, personalized 3-4 sentence mental health summary in English
2. "report_hi": The same report translated to Hindi (Devanagari script)
3. "tips_en": Array of 4-5 actionable self-care tips in English
4. "tips_hi": Array of 4-5 actionable self-care tips in Hindi (Devanagari script)
5. "music_ids": Array of 4-5 calming YouTube video IDs (meditation/relaxing music)
6. "music_titles": Array of titles for each video
7. "breathing_tips": Array of 3 breathing technique names

Be supportive and encouraging. Return ONLY valid JSON, no markdown formatting.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1200
        )
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        return json.loads(content)
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return get_fallback_response(gad_total, gad_percent, voice_score, lang)

def get_fallback_response(gad_total, gad_percent, voice_score, lang):
    """Fallback response when AI is unavailable"""
    severity = "minimal" if gad_total <= 4 else "mild" if gad_total <= 9 else "moderate" if gad_total <= 14 else "significant"
    
    return {
        "report_en": f"Your GAD-7 score is {gad_total}/21 ({gad_percent}%), indicating {severity} anxiety levels. Your voice analysis shows {voice_score}% stress levels. Remember that seeking support is a sign of strength. Consider practicing daily mindfulness and connecting with loved ones.",
        "report_hi": f"आपका GAD-7 स्कोर {gad_total}/21 ({gad_percent}%) है, जो {severity} चिंता का संकेत देता है। आपकी आवाज़ विश्लेषण {voice_score}% तनाव स्तर दिखाता है। कृपया अपनी देखभाल करें और यदि आवश्यक हो तो पेशेवर मदद लें।",
        "tips_en": [
            "Practice deep breathing for 5 minutes daily",
            "Stay hydrated and maintain regular sleep schedule",
            "Take short walks outside when possible",
            "Connect with a friend or family member",
            "Consider professional support if symptoms persist"
        ],
        "tips_hi": [
            "प्रतिदिन 5 मिनट गहरी सांस लेने का अभ्यास करें",
            "पानी पिएं और नियमित नींद का समय बनाए रखें",
            "संभव हो तो बाहर थोड़ा टहलें",
            "किसी मित्र या परिवार के सदस्य से बात करें",
            "यदि लक्षण बने रहें तो पेशेवर सहायता लें"
        ],
        "music_ids": ["lTRiuFIWV54", "1ZYbU82GVz4", "2OEL4P1Rz04", "bX3mZbGBXjg"],
        "music_titles": ["Peaceful Piano Music", "Calm Ocean Waves", "Deep Meditation Sounds", "Relaxing Nature Ambience"],
        "breathing_tips": ["4-7-8 breathing", "Box breathing", "Diaphragmatic breathing"]
    }

# ---------------- ROUTES -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/voice", methods=["POST"])
def api_voice():
    f = request.files.get("file")
    lang = request.form.get("lang","en")
    if not f: return jsonify({"error":"file missing"}),400
    wav = save_webm_to_wav(f.read())
    try:
        features = extract_features(wav).reshape(1,-1)
        machine_label = int(MODEL.predict(features)[0])
        conf = float(np.max(MODEL.predict_proba(features)[0]))
    except:
        machine_label = 0; conf=0.0
    try:
        r = sr.Recognizer()
        with sr.AudioFile(wav) as src:
            audio = r.record(src)
            transcript = r.recognize_google(audio, language="hi-IN" if lang=="hi" else "en-IN")
    except:
        transcript=""
    text_score, found = analyze_text(transcript, lang)
    voice_percent = int(round(0.6*(conf*100)+0.4*text_score))
    os.unlink(wav)
    return jsonify({
        "voice_score":voice_percent,
        "machine_label":machine_label,
        "confidence":conf,
        "transcript":transcript,
        "detected":found
    })

@app.route("/api/gad", methods=["POST"])
def api_gad():
    data = request.get_json() or {}
    answers = data.get("answers",[])
    lang = data.get("lang","en")
    voice_percent = int(data.get("voice_percent",0))
    transcript = data.get("transcript","")
    detected = data.get("detected",[])
    if len(answers)!=7: return jsonify({"error":"need 7 answers"}),400
    
    total, percent, label_map = gad7_score(answers)
    overall = int(round(0.5*percent + 0.5*voice_percent))
    
    spider_chart = create_spider_chart({
        "Voice":voice_percent,
        "GAD-7":percent,
        "Overall":overall,
        "Overthinking":voice_percent//2+percent//2,
        "Fatigue":percent//2+voice_percent//2
    })
    
    ai_resp = ai_analysis(transcript, detected, total, percent, voice_percent, lang)
    
    return jsonify({
        "gad_total":total,
        "gad_percent":percent,
        "gad_label_en":label_map.get("en",""),
        "gad_label_hi":label_map.get("hi",""),
        "voice_score":voice_percent,
        "overall":overall,
        "chart":spider_chart,
        **ai_resp
    })

if __name__=="__main__":
    if not OPENAI_API_KEY:
        print("⚠️  Warning: OPENAI_API_KEY not set. AI analysis will use fallback responses.")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))