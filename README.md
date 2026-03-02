AI-Powered Anxiety Detection System

Overview
This project is a web-based Anxiety Detection System designed to assess mental well-being using a multi-dimensional approach. Instead of relying on a single metric, the application combines structured psychological assessment with machine learning and AI-driven analysis to provide a more comprehensive understanding of anxiety levels.
The system integrates GAD-7 questionnaire responses, voice-based emotional signal analysis, and optional text sentiment interpretation to generate meaningful and interpretable feedback.

Key Features
* GAD-7 questionnaire-based anxiety scoring
* Voice input analysis using extracted audio features
* Text sentiment analysis using VADER
* AI-generated interpretative feedback via OpenAI API
* Machine learning classification using Random Forest
* Web interface built with Flask

Technology Stack
* Python
* Flask
* Scikit-learn
* Librosa
* SpeechRecognition
* VADER Sentiment Analysis
* OpenAI API
* Matplotlib

System Workflow
1. The user submits responses to the GAD-7 questionnaire.
2. Optional voice input is processed to extract emotional features.
3. Optional text input is analyzed for sentiment and contextual meaning.
4. A machine learning model evaluates structured inputs.
5. AI-driven interpretation provides contextual feedback.

The final output reflects a combined assessment rather than a single isolated score.

Installation and Setup

Clone the repository:
git clone <repository-link>
cd AnxietyApp


Create and activate a virtual environment:
Mac/Linux:
python3 -m venv venv
source venv/bin/activate

Windows:
python -m venv venv
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Run the application:
python app.py

Open in browser:
http://127.0.0.1:5000

Security Note:
Ensure that API keys (such as the OpenAI API key) are stored securely using environment variables and are not hardcoded in the source code before making the repository public.

Project Objective:
The goal of this project was to explore how artificial intelligence and machine learning can be applied in a sensitive and human-centered domain such as mental health. It reflects an effort to combine technical implementation with meaningful real-world application, emphasizing interpretability, structured assessment, and responsible AI integration.

