# RespireX: AI-Powered Lung Disease Detection

![RespireX Logo](https://github.com/user-attachments/assets/4856a61b-544d-473b-a3ff-c5af51471dcd)

## Overview

RespireX is an AI-driven healthcare application that leverages deep learning to detect lung diseases from chest X-rays with high accuracy. The platform provides an accessible, affordable alternative to traditional diagnostic methods while reducing dependency on specialist availability.

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Frontend Components](#frontend-components)
- [Machine Learning Models](#machine-learning-models)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Problem Statement

- **Late Diagnosis:** Many lung diseases, including cancer, are detected too late, reducing treatment success rates.
- **Specialist Shortage:** Limited radiologists serving thousands of patients leads to diagnostic delays.
- **High Costs:** Advanced imaging (CT scans, biopsies) are costly, making early detection inaccessible for many populations.
- **Human Error:** Misdiagnosis rates for chest X-rays are significant, leading to incorrect or delayed treatments.

---

## ✅ Solution

RespireX addresses these challenges by:
- **AI-Powered Analysis:** Uses deep learning models to analyze chest X-rays within seconds with high accuracy.
- **User-Friendly Dashboard:** Interactive interface for doctors and patients to access and manage results.
- **Scalable Architecture:** Affordable, cloud-ready deployment model for widespread accessibility.
- **Multi-Level Analysis:** Provides both quick scanning and detailed diagnostic insights.
- **Doctor Integration:** Connects patients with verified medical professionals for consultations.

---

## ⭐ Features

### For Patients
- Upload chest X-rays via web interface
- Receive instant AI-generated diagnoses
- View detailed analysis reports with visualizations
- Access medical history and past results
- Connect with verified doctors for consultations
- Subscription tiers: Free, Premium, Enterprise

### For Doctors
- View and manage patient analysis results
- Provide professional insights on AI predictions
- Track patient cases and ratings
- Access analytics dashboard
- Manage doctor profile and credentials

### Core Features
- **SymptoScan:** Analyzes symptoms and risk factors
- **XrayScan:** Deep learning-based chest X-ray analysis
- **Gemini AI Integration:** Enhanced analysis using Google's Gemini API
- **Multi-Disease Detection:** Identifies multiple lung conditions simultaneously
- **Secure Authentication:** Token-based API authentication

---

## 🛠 Technology Stack

### Backend
- **Framework:** Django 5.1.7 (Python)
- **API:** Django REST Framework with Token Authentication
- **Database:** SQLite (Development), upgradable to PostgreSQL
- **ML/AI:** TensorFlow 2.13, Keras, Scikit-learn
- **Additional Libraries:** NumPy, Pandas, Matplotlib, Seaborn

### Frontend
- **Framework:** React 19 with Vite
- **Routing:** React Router v7
- **Styling:** Tailwind CSS
- **HTTP Client:** Axios
- **Charts:** Recharts
- **Icons:** Lucide React
- **Markdown Support:** Marked.js

### Machine Learning
- **TensorFlow/Keras:** Deep learning model training and inference
- **Models:** 
  - Lung Cancer Classification Model (CNN)
  - Multi-Disease Detection Model
  - XrayScan specialized model

### APIs
- **Gemini AI API:** Enhanced medical analysis insights

---

## 📁 Project Structure

```
RespireX/
├── Backend/                          # Django Backend
│   ├── Backend/                      # Project settings
│   │   ├── settings.py              # Django configuration
│   │   ├── urls.py                  # URL routing
│   │   ├── asgi.py / wsgi.py        # Server configs
│   │   └── __init__.py
│   ├── RespireX_User/               # User management app
│   │   ├── models.py                # UserProfile, APIUser models
│   │   ├── views.py                 # API endpoints
│   │   ├── urls.py                  # User routes
│   │   ├── GeminiAnalyze.py         # Gemini integration
│   │   ├── admin.py
│   │   ├── migrations/              # Database migrations
│   │   └── __pycache__/
│   ├── Doctor/                      # Doctor management app
│   │   ├── models.py                # Doctor model
│   │   ├── views.py                 # Doctor endpoints
│   │   ├── urls.py                  # Doctor routes
│   │   ├── admin.py
│   │   ├── migrations/
│   │   └── __pycache__/
│   ├── Model_Training/              # ML model training & inference
│   │   ├── main_code.py            # Primary model training pipeline
│   │   ├── SymptoScan.py           # Symptom-based analysis
│   │   ├── XrayScan.py             # X-ray image analysis
│   │   ├── Dataset/                # Training datasets
│   │   │   ├── survey.csv          # Symptom survey data
│   │   │   ├── Data_Entry_2017.csv # Medical records
│   │   │   ├── 01test.csv          # Test dataset
│   │   │   └── predictions_output.csv
│   │   ├── model/                  # Pre-trained models
│   │   │   ├── lung_cancer_model.h5
│   │   │   ├── multi_disease_model.json
│   │   │   └── team7_model.h5
│   │   ├── Notebook/               # Jupyter notebooks
│   │   │   ├── cnn_all_notebook.ipynb
│   │   │   └── Lung_Cancer_Classify_Model_.ipynb
│   │   ├── test/                   # Testing scripts
│   │   │   └── 01test.py
│   │   └── __pycache__/
│   ├── manage.py                    # Django management
│   ├── db.sqlite3                   # Database
│   └── requirements.txt             # Python dependencies
├── Frontend/                        # React Frontend
│   ├── src/
│   │   ├── App.jsx                 # Main app component
│   │   ├── main.jsx                # Entry point
│   │   ├── index.css               # Global styles
│   │   ├── components/
│   │   │   ├── Navbar.jsx          # Navigation
│   │   │   ├── Home.jsx            # Landing page
│   │   │   ├── Auth.jsx            # Login/Signup
│   │   │   ├── XrayAnalysis.jsx    # X-ray upload & analysis
│   │   │   ├── PatientAnalysis.jsx # Patient results
│   │   │   ├── ApiHistory.jsx      # User history
│   │   │   ├── ResultsDashboard.jsx# Results visualization
│   │   │   ├── Features.jsx        # Features showcase
│   │   │   └── upgrade.jsx         # Subscription tiers
│   │   ├── assets/
│   │   │   ├── icons/              # Icon assets
│   │   │   └── images/             # Image assets
│   ├── package.json
│   ├── vite.config.js              # Vite configuration
│   ├── tailwind.config.js          # Tailwind CSS config
│   ├── postcss.config.js
│   ├── eslint.config.js
│   ├── index.html
│   └── public/
├── Documentation/                  # Project documentation
├── README.md                        # This file
└── requirements.txt               # Python dependencies
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git
- pip and npm package managers

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/tusharneje-07/RespireX.git
cd RespireX
```

2. Navigate to Backend directory:
```bash
cd Backend
```

3. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Apply database migrations:
```bash
python manage.py migrate
```

6. Create superuser (admin account):
```bash
python manage.py createsuperuser
```

7. Start Django development server:
```bash
python manage.py runserver 0.0.0.0:8000
```

Backend will be available at `http://127.0.0.1:8000`

### Frontend Setup

1. Navigate to Frontend directory:
```bash
cd Frontend
```

2. Install Node dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

Frontend will be available at `http://localhost:5173`

### Environment Configuration

**Backend Settings** (`Backend/Backend/settings.py`):
- `ALLOWED_HOSTS`: Configure for your deployment
- `DEBUG`: Set to `False` in production
- `SECRET_KEY`: Change in production
- `DATABASES`: Configure PostgreSQL for production

---

## 📡 API Documentation

### Authentication
- **Type:** Token-based authentication
- **Header:** `Authorization: Token <auth_token>`
- **Token Generation:** Automatically generated on user creation

### User Endpoints

#### 1. User Login
```
POST /api/user/login/
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}

Response:
{
  "token": "sha256_hash_token",
  "user_id": 1,
  "message": "Login successful"
}
```

#### 2. X-Ray Analysis (Level 0 - Quick Scan)
```
POST /api/analysis/level0/
Authorization: Token <auth_token>
Content-Type: multipart/form-data

{
  "image": <image_file>,
  "model_name": "xray" | "lung_cancer"
}

Response:
{
  "prediction": "Tuberculosis",
  "confidence": 0.92,
  "timestamp": "2025-01-14T10:30:00Z",
  "analysis_id": "abc123"
}
```

#### 3. Detailed Analysis (Level 1)
```
POST /api/analysis/level1/
Authorization: Token <auth_token>
Content-Type: multipart/form-data

{
  "image": <image_file>,
  "user_symptoms": ["cough", "fever"],
  "medical_history": "asthma"
}

Response:
{
  "diseases_detected": [
    {
      "name": "Tuberculosis",
      "confidence": 0.92,
      "severity": "High"
    }
  ],
  "gemini_analysis": "Detailed insights from Gemini AI",
  "recommendations": ["Consult specialist", "Get blood test"],
  "timestamp": "2025-01-14T10:30:00Z"
}
```

#### 4. Get User History
```
GET /api/user/history/
Authorization: Token <auth_token>

Response:
{
  "total_scans": 5,
  "history": [
    {
      "id": 1,
      "image_url": "/media/xray_001.jpg",
      "prediction": "Pneumonia",
      "date": "2025-01-14T10:30:00Z"
    }
  ]
}
```

#### 5. Get Doctor Details
```
GET /api/doctor/details/?doctor_id=1
Authorization: Token <auth_token>

Response:
{
  "id": 1,
  "username": "dr_smith",
  "speciality": "Pulmonology",
  "experience": 15,
  "cases_handled": 500,
  "rating": 4.8
}
```

---

## 🗄️ Database Schema

### UserProfile Model
```
- id (PK)
- user (FK → User)
- phone_number (CharField)
- full_name (CharField)
- gender (Choice: M/F/N)
- age (PositiveIntegerField)
- created_at (auto_now_add)
```

### APIUser Model
```
- id (PK)
- user (FK → User)
- account_type (Choice: Free/Premium/Enterprise)
- total_hits (PositiveIntegerField)
- up_and_running_date (DateTimeField)
- history (JSONField)
- auth_token (CharField, unique)
- is_active (BooleanField)
```

### Doctor Model
```
- id (PK)
- username (CharField, unique)
- speciality (CharField)
- keywords (TextField)
- experience (PositiveIntegerField)
- cases_handled (PositiveIntegerField)
- rating (DecimalField, max 5.0)
```

---

## 🎨 Frontend Components

### Core Components
- **Navbar.jsx:** Navigation bar with user menu
- **Auth.jsx:** Login and registration forms
- **Home.jsx:** Landing page with features overview
- **XrayAnalysis.jsx:** X-ray upload and live analysis
- **PatientAnalysis.jsx:** Patient-specific analysis results
- **ResultsDashboard.jsx:** Visualization of medical data

### Pages
- **Home:** Introduction and features showcase
- **Analysis:** Upload and analyze X-rays
- **History:** View past scans and results
- **Doctor Connection:** Find and connect with specialists
- **Account:** User profile and subscription management

---

## 🤖 Machine Learning Models

### 1. Lung Cancer Classification Model
- **File:** `lung_cancer_model.h5`
- **Type:** CNN (Convolutional Neural Network)
- **Training Data:** Chest X-ray images
- **Output:** Binary classification (Cancer/No Cancer)
- **Accuracy:** ~92%

### 2. Multi-Disease Detection Model
- **File:** `multi_disease_model.json`
- **Type:** Deep Learning Model
- **Diseases Detected:** Tuberculosis, Pneumonia, COVID-19, Asthma, etc.
- **Input:** Chest X-ray image
- **Output:** Multiple disease probabilities

### 3. Team7 Specialized Model
- **File:** `team7_model.h5`
- **Purpose:** Specialized analysis for specific conditions

### Model Training Process
1. Data preprocessing and normalization
2. Train-test split (80-20)
3. CNN architecture with multiple convolutional layers
4. Adam optimizer and categorical crossentropy loss
5. ROC-AUC evaluation

---

## 📊 Key Features Implementation

### SymptoScan Module
- Analyzes patient symptoms and medical history
- Predicts disease probability based on symptoms
- Integrates with X-ray analysis for comprehensive diagnosis

### XrayScan Module
- Processes chest X-ray images
- Applies preprocessing (resizing, normalization)
- Runs inference on trained models
- Returns disease predictions with confidence scores

### Gemini Integration
- Enhances AI predictions with contextual analysis
- Provides personalized health recommendations
- Generates detailed medical insights
- Supports natural language queries

---

## 🔒 Security Features

- **Token Authentication:** Secure API access with unique tokens
- **CORS Enabled:** Controlled cross-origin access
- **Encrypted Passwords:** Django's built-in password hashing
- **API Rate Limiting:** (Can be implemented)
- **Database Security:** SQLite in development, PostgreSQL recommended for production

---

## 🚀 Deployment

### Production Checklist
- [ ] Set `DEBUG = False` in settings.py
- [ ] Update `SECRET_KEY` with a strong random value
- [ ] Configure `ALLOWED_HOSTS` for your domain
- [ ] Migrate to PostgreSQL database
- [ ] Set up environment variables for sensitive data
- [ ] Enable HTTPS/SSL
- [ ] Configure static file serving
- [ ] Set up logging and monitoring

### Recommended Deployment Platforms
- **Backend:** Heroku, AWS EC2, DigitalOcean, Azure
- **Frontend:** Vercel, Netlify, GitHub Pages
- **Database:** PostgreSQL on managed service
- **Storage:** AWS S3 for medical images

### Environment Variables Example
```
# Backend
DEBUG=False
SECRET_KEY=your_secret_key_here
DATABASE_URL=postgresql://user:password@host:port/dbname
GEMINI_API_KEY=your_gemini_api_key
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Frontend
VITE_API_URL=https://api.yourdomain.com
```

---

## 👥 User Account Types

### Free Tier
- 5 X-ray scans/month
- Basic analysis
- Limited history

### Premium Tier
- 50 X-ray scans/month
- Enhanced Gemini analysis
- Priority doctor consultations
- Full history retention

### Enterprise Tier
- Unlimited scans
- Custom integrations
- Dedicated support
- API access for clinics/hospitals

---

## 📝 API Usage Examples

### Python Example
```python
import requests

# Login
response = requests.post('http://localhost:8000/api/user/login/', {
    'username': 'user@example.com',
    'password': 'password123'
})
token = response.json()['token']

# Upload X-ray
headers = {'Authorization': f'Token {token}'}
files = {'image': open('xray.jpg', 'rb')}
response = requests.post(
    'http://localhost:8000/api/analysis/level0/',
    headers=headers,
    files=files,
    data={'model_name': 'xray'}
)
print(response.json())
```

### JavaScript/React Example
```javascript
// Login
const loginResponse = await fetch('http://localhost:8000/api/user/login/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'user@example.com', password: 'password123' })
});
const { token } = await loginResponse.json();

// Upload X-ray
const formData = new FormData();
formData.append('image', xrayFile);
formData.append('model_name', 'xray');

const analysisResponse = await fetch('http://localhost:8000/api/analysis/level0/', {
  method: 'POST',
  headers: { 'Authorization': `Token ${token}` },
  body: formData
});
const result = await analysisResponse.json();
console.log(result);
```

---

## 🐛 Troubleshooting

### Backend Issues

**Issue:** Django migrations fail
```bash
# Solution
python manage.py makemigrations
python manage.py migrate --run-syncdb
```

**Issue:** CORS errors when connecting frontend
- Ensure `corsheaders` is in `INSTALLED_APPS`
- Add `'corsheaders.middleware.CorsMiddleware'` to `MIDDLEWARE`
- Configure `CORS_ALLOWED_ORIGINS` in settings

**Issue:** Model file not found
- Verify model files exist in `Backend/Model_Training/model/`
- Check file paths in analysis code

### Frontend Issues

**Issue:** API requests failing
- Verify backend is running on `http://localhost:8000`
- Check `VITE_API_URL` environment variable
- Ensure token is included in Authorization header

**Issue:** Page blank after build
- Run `npm run build` to rebuild
- Check browser console for errors
- Verify Vite configuration

---

## 📚 Documentation Files

- **[Backend API Docs](Backend/README.md):** Detailed backend documentation
- **[Frontend Setup](Frontend/README.md):** Frontend development guide
- **[Model Training Guide](Backend/Model_Training/README.md):** ML model training documentation

---

## 👥 Team

- **Tushar Neje**
- **Irfan Naikwade**
- **Sandesh Pol**
- **Sushant Khadake**

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a Pull Request

### Code Guidelines
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions
- Test your changes before submitting

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 📞 Support & Contact

For support, issues, or inquiries:
- **Email:** support@respirex.com
- **Issues:** GitHub Issues page
- **Documentation:** [Full Wiki](https://github.com/tusharneje-07/RespireX/wiki)

---

## 🎯 Roadmap

- [ ] Mobile app (iOS/Android)
- [ ] Multi-language support
- [ ] Enhanced Gemini AI integration
- [ ] Real-time collaboration features
- [ ] Video consultation support
- [ ] Blockchain-based medical records
- [ ] Advanced analytics dashboard

---

**Last Updated:** January 14, 2026  
**Version:** 1.0.0
