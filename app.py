from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import os
import csv

# Initialize Flask application
app = Flask(__name__)

# Load dataset with health provider information
health_provider_info = pd.read_csv("datasets/Doctors_and_Diseases.csv")

# Load pre-trained SVM classifier from a serialized file
svc = pickle.load(open("models/svc.pkl", 'rb'))

#==================================================
# Symptoms Dictionary and DiseaseList Dictionary
#==================================================
symptom_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 117, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_ofurine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131, 'prognosis': 132}
disease_list_dict = {0: '(vertigo) Paroymsal Positional Vertigo', 1: 'AIDS', 2: 'Acne', 3: 'Alcoholic hepatitis', 4: 'Allergy', 5: 'Arthritis', 6: 'Bronchial Asthma', 7: 'Cervical spondylosis', 8: 'Chicken pox', 9: 'Chronic cholestasis', 10: 'Common Cold', 11: 'Dengue', 12: 'Diabetes', 13: 'Dimorphic hemmorhoids(piles)', 14: 'Drug Reaction', 15: 'Fungal infection', 16: 'GERD', 17: 'Gastroenteritis', 18: 'Heart attack', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 23: 'Hypertension', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 26: 'Hypothyroidism', 27: 'Impetigo', 28: 'Jaundice', 29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthristis', 32: 'Paralysis (brain hemorrhage)', 33: 'Peptic ulcer diseae', 34: 'Pneumonia', 35: 'Psoriasis', 36: 'Tuberculosis', 37: 'Typhoid', 38: 'Urinary tract infection', 39: 'Varicose veins', 40: 'hepatitis A'}

# All unique disease names available in the dataset
all_diseases = health_provider_info['Disease'].dropna().unique().tolist()

# Common aliases/mappings for symptoms and disease shorthand
symptom_aliases = {
    'fever': 'high_fever',
    'cold': 'continuous_sneezing',
    'sneezing': 'continuous_sneezing',
    'rash': 'skin_rash',
    'joint pain': 'joint_pain',
    'stomach pain': 'stomach_pain',
    'chest pain': 'chest_pain',
    'breath problem': 'breathlessness',
    'vomit': 'vomiting',
}

disease_aliases = {
    'asthma': 'Bronchial Asthma',
    'piles': 'Dimorphic hemmorhoids(piles)',
    'vertigo': '(vertigo) Paroymsal Positional Vertigo',
    'peptic ulcer': 'Peptic ulcer diseae',
    'osteoarthritis': 'Osteoarthristis',
    'sugar': 'Diabetes',
    'bp': 'Hypertension',
    'high bp': 'Hypertension',
}

#======= model prediction function =======
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptom_dict))
    for item in patient_symptoms:
        if item in symptom_dict:
            input_vector[symptom_dict[item]] = 1
    return disease_list_dict[svc.predict([input_vector])[0]]

#======= identify disease from input or predict from symptoms =======
def identify_disease_or_predict(user_input):
    if not user_input or not user_input.strip():
        return None, "Please enter a disease name or symptoms."

    cleaned = user_input.strip()
    cleaned_lower = cleaned.lower()

    # 1. Check known disease aliases
    if cleaned_lower in disease_aliases:
        return disease_aliases[cleaned_lower], None

    # 2. Check exact case-insensitive match against all diseases
    for d in all_diseases:
        if d.lower() == cleaned_lower:
            return d, None

    # 3. Check substring match (e.g. user typed 'diabetes' or 'fungal')
    for d in all_diseases:
        if cleaned_lower in d.lower() or d.lower() in cleaned_lower:
            return d, None

    # 4. Check if the input is a list of symptoms
    raw_parts = [s.strip().lower() for s in cleaned.split(',') if s.strip()]
    if not raw_parts:
        return None, "Please enter valid symptoms or a disease name."

    # Normalize symptoms and apply aliases
    normalized_symptoms = []
    for s in raw_parts:
        s_clean = s.replace('-', '_').replace(' ', '_')
        if s in symptom_aliases:
            s_clean = symptom_aliases[s]
        elif s.replace('_', ' ') in symptom_aliases:
            s_clean = symptom_aliases[s.replace('_', ' ')]
        normalized_symptoms.append(s_clean)

    # Check for invalid symptoms
    invalid = [s for s in normalized_symptoms if s not in symptom_dict]
    if invalid:
        # If any token matches a disease name
        for inv in invalid:
            inv_plain = inv.replace('_', ' ')
            if inv_plain in disease_aliases:
                return disease_aliases[inv_plain], None
            for d in all_diseases:
                if inv_plain in d.lower() or d.lower() in inv_plain:
                    return d, None

        return None, f"Unrecognized input: {', '.join(invalid)}. Please enter a disease name (e.g. Diabetes, Malaria) or recognized symptoms (e.g. chills, shivering, cough)."

    predicted_disease = get_predicted_value(normalized_symptoms)
    return predicted_disease, None

#======= doctor recommendation function with smart fallbacks =======
def get_doctor_recommendations(predicted_disease, specified_time=None, location=None):
    doctors = health_provider_info[health_provider_info['Disease'] == predicted_disease].copy()
    if doctors.empty:
        return [], f"No doctor records found in the database for {predicted_disease}."

    notice = ""

    # Helper function to match city / state
    def filter_by_location(df_in, loc_str):
        if not loc_str:
            return df_in
        loc_str = loc_str.strip().lower()
        if ',' in loc_str:
            parts = [p.strip() for p in loc_str.split(',') if p.strip()]
            if len(parts) >= 2:
                m = df_in[(df_in['City'].str.lower().str.contains(parts[0], regex=False)) &
                          (df_in['State'].str.lower().str.contains(parts[1], regex=False))]
                if not m.empty:
                    return m
            for p in parts:
                m = df_in[(df_in['City'].str.lower().str.contains(p, regex=False)) |
                          (df_in['State'].str.lower().str.contains(p, regex=False))]
                if not m.empty:
                    return m
        else:
            m = df_in[(df_in['City'].str.lower().str.contains(loc_str, regex=False)) |
                      (df_in['State'].str.lower().str.contains(loc_str, regex=False))]
            if not m.empty:
                return m
        return pd.DataFrame()

    # Helper function to check if time falls in doctor slot
    def is_time_within_slots(slots_str, check_time):
        for slot in str(slots_str).split(','):
            parts = slot.split('-')
            if len(parts) == 2:
                try:
                    start_time = datetime.strptime(parts[0].strip(), "%H:%M").time()
                    end_time = datetime.strptime(parts[1].strip(), "%H:%M").time()
                    if start_time <= check_time <= end_time:
                        return True
                except Exception:
                    pass
        return False

    time_obj = None
    if specified_time:
        try:
            time_obj = datetime.strptime(specified_time.strip(), "%H:%M").time()
        except Exception:
            pass

    # Tier 1 & 2: Match Location if provided
    matched_loc = filter_by_location(doctors, location) if location else doctors

    if not matched_loc.empty:
        if time_obj:
            matched_time = matched_loc[matched_loc['Available Time Slots'].apply(
                lambda x: is_time_within_slots(x, time_obj))]
            if not matched_time.empty:
                doctors = matched_time
                notice = f"Showing doctors available in {location} around {specified_time}."
            else:
                doctors = matched_loc
                notice = f"No doctor free at exactly {specified_time} in {location}. Showing available doctors in {location} with their respective time slots."
        else:
            doctors = matched_loc
            if location:
                notice = f"Showing doctors available in {location}."
    else:
        # If no doctors found in that specific location, fall back to all doctors for the disease
        if time_obj:
            matched_time = doctors[doctors['Available Time Slots'].apply(
                lambda x: is_time_within_slots(x, time_obj))]
            if not matched_time.empty:
                doctors = matched_time
                notice = f"No doctor found specifically in \"{location}\" for {predicted_disease}. Showing doctors available around {specified_time} in other cities."
            else:
                notice = f"No doctor found specifically in \"{location}\" for {predicted_disease}. Showing all top-rated doctors for this condition."
        else:
            notice = f"No doctor found specifically in \"{location}\" for {predicted_disease}. Showing all top-rated doctors for this condition."

    # Sort doctors by User Rating in descending order
    doctors = doctors.sort_values(by='User Rating', ascending=False)
    results = doctors[['Doctor Name', 'City', 'State', 'Available Time Slots', 'User Rating', 'Hospital']].to_dict(orient='records')
    return results, notice

# Path for storing user feedback
FEEDBACK_FILE = os.path.join("datasets", "feedback.csv")

#======= Routes =======
@app.route('/')
def index():
    symptoms_list = list(symptom_dict.keys())
    return render_template('index.html', symptoms_list=symptoms_list)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json(silent=True) or request.form
    feedback_choice = data.get('feedback', '').strip()
    comments = data.get('comments', '').strip()

    if not feedback_choice or feedback_choice == "Please choose an option":
        return jsonify({"status": "error", "message": "Please select a valid feedback option."}), 400

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure datasets directory exists and write feedback to CSV file
    os.makedirs("datasets", exist_ok=True)
    file_exists = os.path.isfile(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Feedback", "Comments"])
        writer.writerow([timestamp, feedback_choice, comments])

    return jsonify({"status": "success", "message": "Thank you! Your feedback has been saved successfully."})

@app.route('/feedbacks')
def view_feedbacks():
    feedbacks_list = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            df_fb = pd.read_csv(FEEDBACK_FILE)
            feedbacks_list = df_fb.fillna('').to_dict(orient='records')
            feedbacks_list.reverse()  # Show latest feedback first
        except Exception:
            pass
    return render_template('feedbacks.html', feedbacks=feedbacks_list)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()
        specified_time = request.form.get('time', '').strip()
        city_state = request.form.get('city_state', '').strip()

        if not symptoms:
            return render_template(
                'index.html',
                error="Please enter a disease name or symptoms.",
                input_symptoms=symptoms,
                input_time=specified_time,
                input_city_state=city_state
            )

        predicted_disease, error_message = identify_disease_or_predict(symptoms)

        if error_message:
            return render_template(
                'index.html',
                error=error_message,
                input_symptoms=symptoms,
                input_time=specified_time,
                input_city_state=city_state
            )

        available_doctors, filter_notice = get_doctor_recommendations(
            predicted_disease,
            specified_time=specified_time if specified_time else None,
            location=city_state if city_state else None
        )

        return render_template(
            'index.html',
            predicted_disease=predicted_disease,
            health_providers=available_doctors,
            filter_notice=filter_notice,
            input_symptoms=symptoms,
            input_time=specified_time,
            input_city_state=city_state
        )

    return render_template('index.html')

# ===============main python============ #
if __name__ == "__main__":
    app.run(debug=True)
