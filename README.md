# CareQuest - A Healthcare Recommendation System

## 📜 Table of Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)
- [Setup and Launch](#-setup-and-launch)
- [Feedback & Contact](#-feedback--contact)

## 🌟 Introduction

This is a Flask-based web application that offers healthcare recommendations based on user symptoms or disease name. It utilizes machine learning algorithms to predict potential health conditions and recommends specialized doctors for further consultation.

The system supports:
- **Disease name input** (e.g. Diabetes, Malaria, Fungal infection)
- **Symptom-based prediction** (e.g. chills, shivering, cough)
- **Location-based filtering** (City, State)
- **Time-slot based filtering** for doctor appointments

## 🔥 Features
- Predicts diseases based on user-entered symptoms using an SVM (Support Vector Machine) model
- Also accepts direct disease name input for instant doctor recommendations
- Recommends specialized doctors for identified diseases, sorted by user rating
- **Location-Based Search**: Search for doctors by city or state
- **Time-Slot Filter**: Find doctors available at a preferred time
- **Smart Fallback**: If no doctor found in selected city, shows top doctors from other cities
- **Feedback Mechanism**: Users can submit feedback saved to `datasets/feedback.csv`
- **Feedback Dashboard**: Admin can view all feedbacks at `/feedbacks`
- Modern, responsive UI with video slideshow and glassmorphic design

## 💡 Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn (SVM Classifier)
- **Data Handling**: Pandas, NumPy
- **Frontend**: HTML5, CSS3 (custom glassmorphic design), JavaScript (Fetch API)
- **UI Framework**: Bootstrap 5.3

## 🚀 Getting Started

1. **Access the Portal**: Open `http://localhost:5000` after running the app locally.
2. **Enter Disease or Symptoms**: Type a disease name (e.g. `Diabetes`) or comma-separated symptoms (e.g. `chills, shivering, cough`).
3. **Optional - Set Preferred Time**: Select a time to filter doctors available at that hour.
4. **Optional - Set Location**: Enter your city or state (e.g. `Delhi`, `Mumbai`, `Maharashtra`).
5. **Generate Recommendations**: Click the button to get top-rated doctors instantly.
6. **View Results**: Disease name and doctor list will appear below the form.
7. **Submit Feedback**: Share your experience using the feedback form at the bottom.
8. **View All Feedbacks**: Visit `http://localhost:5000/feedbacks` to see all submitted feedback.

## 📦 Dependencies

Ensure your environment has the following packages:

```
Flask
pandas
numpy
scikit-learn
```

Install all at once:
```bash
pip install -r requirements.txt
```

## 💼 Setup and Launch

Follow these steps to run CareQuest locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/satyamgit77/CareQuest.git
   cd CareQuest
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model** (optional, model already included):
   ```bash
   python train_model.py
   ```

5. **Run the Application**:
   ```bash
   python app.py
   ```

6. Open your browser at `http://localhost:5000`

## ✉ Feedback & Contact

We value your feedback and suggestions for improving CareQuest. Feel free to reach out for any questions, feedback, or collaboration opportunities:

- **Developer**: Satyam
- **GitHub**: [github.com/satyamgit77](https://github.com/satyamgit77)
- **Email**: rasatyam661@gmail.com
