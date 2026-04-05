This project can be installed and executed by setting up both the backend (Flask + BERT model) and frontend (React dashboard) environments. First, clone the repository and create a Python virtual environment, then install all required dependencies using pip install -r requirements.txt. Configure essential credentials including Google OAuth (credentials.json), Firebase service account key (firebase_key.json), and environment variables such as SECRET_KEY and HF_TOKEN. Once configured, start the Flask backend using python app.py, which runs on port 7860 and automatically downloads and loads the fine-tuned BERT model from HuggingFace. Next, navigate to the frontend directory, install Node dependencies using npm install, update the API URL if necessary, and start the React application with npm start. After launching, connect Gmail via the /login route to enable real-time email monitoring. Once setup is complete, the system fetches emails, performs phishing detection using BERT, generates LIME explanations, stores results in Firebase, and displays them on the dashboard.

 System Requirements
Python 3.9+
Node.js 18+
npm or yarn
Google Cloud Project
Firebase Project
HuggingFace account


STEP 1: Clone Repository
git clone https://github.com/your-repo/email-detection.git
cd email-detection

🔥 STEP 2: Backend Setup (Flask)
2.1 Create Virtual Environment
python -m venv venv

Activate:

Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
2.2 Install Dependencies

Create requirements.txt:

flask
flask-cors
google-auth
google-auth-oauthlib
google-api-python-client
firebase-admin
transformers
torch
lime
beautifulsoup4
huggingface_hub
psutil
numpy

Then install:

pip install -r requirements.txt
🔥 STEP 3: Google Cloud Setup
3.1 Enable APIs

Enable:

Gmail API
Pub/Sub API
3.2 Create OAuth Credentials
Go to Google Cloud Console
Create OAuth 2.0 Client ID
Add redirect URI:
http://localhost:7860/oauth2callback
Download credentials.json
Place inside project root
🔥 STEP 4: Firebase Setup
4.1 Create Firebase Project
Go to Firebase Console
Create new project
Enable Firestore Database
4.2 Generate Service Account Key
Project Settings
Service Accounts
Generate new private key
Download JSON

Rename to:

firebase_key.json

Place in root folder.

🔥 STEP 5: HuggingFace Setup
5.1 Create HuggingFace Token
Go to https://huggingface.co/settings/tokens
Generate Access Token
5.2 Set Environment Variable

Windows:

set HF_TOKEN=your_token_here

Mac/Linux:

export HF_TOKEN=your_token_here
🔥 STEP 6: Environment Variables

Create .env file:

SECRET_KEY=your_secret_key
HF_TOKEN=your_huggingface_token
REDIRECT_URI=http://localhost:7860/oauth2callback
🔥 STEP 7: Run Backend
python app.py

Server runs at:

http://localhost:7860
🔥 STEP 8: Frontend Setup (React)

Navigate to frontend folder:

cd frontend
npm install
8.1 Update API URL

Inside Dashboard.js:

const API_URL = "http://localhost:7860";
8.2 Start React App
npm start

Frontend runs at:

http://localhost:3000
🔐 STEP 9: Connect Gmail

Visit:

http://localhost:7860/login

Grant Gmail permission.

📡 STEP 10: Start Gmail Watch
http://localhost:7860/start-watch

This enables real-time email monitoring.

🤖 Model Loading
BERT model downloads automatically from HuggingFace
Stored in local folder bert_model/
Loaded into CPU memory
Inference runs using PyTorch
📊 API Endpoints
Endpoint	Method	Description
/api/emails	GET	Get all processed emails
/api/low-confidence	GET	Get emails with low confidence
/api/submit-feedback	POST	Submit user correction
/trigger-retraining	POST	Start retraining pipeline
/reload-model	POST	Reload BERT model
/status	GET	System status
/health	GET	Health check
🔁 Retraining Flow
Low confidence emails stored
User submits feedback
Feedback stored in Firestore
If threshold reached (20 per class)
Retraining pipeline triggered
Model updated on HuggingFace
Backend reloads new model
🧠 LIME Explanation