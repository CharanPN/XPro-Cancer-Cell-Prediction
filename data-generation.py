import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import timedelta

# -------------------------
# Data Generation (Clean Dataset)
# -------------------------

fake = Faker()
n = 1000  # number of patient records

def random_blood_group():
    return random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

def random_smoking_status():
    return random.choice(["Never", "Former", "Current"])

def random_alcohol_consumption():
    return random.choice(["None", "Moderate", "High"])

def random_physical_activity():
    return random.choice(["Low", "Moderate", "High"])

def random_diet_quality():
    return random.choice(["Poor", "Average", "Healthy"])

def random_family_history():
    return random.choice(["Yes", "No"])

def random_blood_pressure():
    systolic = random.randint(90, 160)
    diastolic = random.randint(60, 100)
    return f"{systolic}/{diastolic}"

def random_department():
    return random.choice(["Oncology", "Cardiology", "Neurology", "Pediatrics", "General Medicine"])

def random_doctor():
    return f"Dr. {fake.last_name()}"

def random_insurance():
    return random.choice(["MediCare", "HealthPlus", "LifeSecure", "CareFirst", "None"])

data = []
for i in range(1, n+1):
    patient_id = f"P{i:04d}"
    first_name = fake.first_name()
    last_name = fake.last_name()
    age = random.randint(1, 100)
    gender = random.choice(["M", "F"])
    blood_group = random_blood_group()

    smoking = random_smoking_status()
    alcohol = random_alcohol_consumption()
    family_history = random_family_history()
    activity = random_physical_activity()
    diet = random_diet_quality()

    height = random.randint(140, 200)
    weight = random.randint(40, 120)
    bmi = round(weight / ((height/100)**2), 1)
    bp = random_blood_pressure()
    heart_rate = random.randint(60, 100)

    blood_sugar = random.choices(
        [random.randint(70, 100), random.randint(101, 125), random.randint(126, 200)],
        weights=[0.6, 0.25, 0.15]
    )[0]

    cholesterol = random.choices(
        [random.randint(150, 199), random.randint(200, 239), random.randint(240, 300)],
        weights=[0.5, 0.3, 0.2]
    )[0]

    wbc = round(np.random.normal(7.0, 2.0), 1)
    platelets = int(np.random.normal(250, 50))
    hemoglobin = round(np.random.normal(14.0, 1.5), 1)
    tumor_marker = round(np.random.choice(
        [np.random.uniform(0, 35), np.random.uniform(36, 100)],
        p=[0.85, 0.15]
    ), 2)

    doctor = random_doctor()
    admission_date = fake.date_between(start_date="-5y", end_date="today")
    discharge_date = admission_date + timedelta(days=random.randint(1, 30))
    department = random_department()
    insurance = random_insurance()

    data.append([
        patient_id, first_name, last_name, age, gender, blood_group,
        smoking, alcohol, family_history, activity, diet,
        height, weight, bmi, bp, heart_rate, blood_sugar, cholesterol,
        wbc, platelets, hemoglobin, tumor_marker,
        doctor, admission_date, discharge_date, department, insurance
    ])

columns = [
    "Patient_ID", "First_Name", "Last_Name", "Age", "Gender", "Blood_Group",
    "Smoking_Status", "Alcohol_Consumption", "Family_History_Cancer",
    "Physical_Activity", "Diet_Quality",
    "Height_cm", "Weight_kg", "BMI", "Blood_Pressure", "Heart_Rate",
    "Blood_Sugar_mg/dL", "Cholesterol_mg/dL",
    "WBC_Count", "Platelet_Count", "Hemoglobin_g/dL", "Tumor_Marker_Level",
    "Doctor_Assigned", "Admission_Date", "Discharge_Date",
    "Hospital_Department", "Insurance_Provider"
]

df = pd.DataFrame(data, columns=columns)

# -------------------------
# 1. Introduce Anomalies
# -------------------------

df_anomaly = df.copy()

# Missing values
for col in ["Age", "BMI", "Blood_Sugar_mg/dL", "Cholesterol_mg/dL"]:
    df_anomaly.loc[df_anomaly.sample(frac=0.02).index, col] = np.nan

# Duplicates
duplicates = df_anomaly.sample(10, random_state=42)
df_anomaly = pd.concat([df_anomaly, duplicates], ignore_index=True)

# Outliers
df_anomaly.loc[random.sample(range(len(df_anomaly)), 5), "Age"] = 999
df_anomaly.loc[random.sample(range(len(df_anomaly)), 5), "BMI"] = -5

# Wrong formats
df_anomaly.loc[random.sample(range(len(df_anomaly)), 5), "Age"] = "forty"
df_anomaly.loc[random.sample(range(len(df_anomaly)), 5), "Blood_Sugar_mg/dL"] = "high"

print(" Data with anomalies introduced:")
print(df_anomaly.head(10))

# -------------------------
# 2. Clean Data
# -------------------------

df_clean = df_anomaly.copy()

# Remove duplicates
df_clean = df_clean.drop_duplicates()

# Fix data types
numeric_cols = ["Age", "BMI", "Blood_Sugar_mg/dL", "Cholesterol_mg/dL",
                "WBC_Count", "Platelet_Count", "Hemoglobin_g/dL", "Tumor_Marker_Level"]
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# Handle outliers
df_clean["Age"] = df_clean["Age"].clip(lower=0, upper=110)
df_clean["BMI"] = df_clean["BMI"].clip(lower=10, upper=60)
df_clean["Blood_Sugar_mg/dL"] = df_clean["Blood_Sugar_mg/dL"].clip(lower=50, upper=400)
df_clean["Cholesterol_mg/dL"] = df_clean["Cholesterol_mg/dL"].clip(lower=100, upper=400)

# Fill missing values
for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

categorical_cols = ["Gender", "Blood_Group", "Smoking_Status", "Alcohol_Consumption",
                    "Family_History_Cancer", "Physical_Activity", "Diet_Quality",
                    "Hospital_Department", "Insurance_Provider"]
for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print("\n Cleaned dataset preview:")
print(df_clean.head(10))

# -------------------------
# 3. Summary Report
# -------------------------

print("\n Cleaning Summary:")
print(f" - Duplicates added: {len(df_anomaly) - len(df)}")
print(f" - Duplicates removed: {len(df_anomaly) - len(df_clean)}")

print(" - Missing values (before cleaning):")
print(df_anomaly.isna().sum()[df_anomaly.isna().sum() > 0])

print(" - Missing values (after cleaning):", df_clean.isna().sum().sum())

print(f" - Age min/max after cleaning: {df_clean['Age'].min()}/{df_clean['Age'].max()}")
print(f" - BMI min/max after cleaning: {df_clean['BMI'].min()}/{df_clean['BMI'].max()}")
print(f" - Blood Sugar min/max after cleaning: {df_clean['Blood_Sugar_mg/dL'].min()}/{df_clean['Blood_Sugar_mg/dL'].max()}")
print(f" - Cholesterol min/max after cleaning: {df_clean['Cholesterol_mg/dL'].min()}/{df_clean['Cholesterol_mg/dL'].max()}")

# -------------------------
# 4. Save files
# -------------------------

df_anomaly.to_excel("synthetic_patient_data_with_anomalies.xlsx", index=False)
df_clean.to_excel("synthetic_patient_data_cleaned.xlsx", index=False)

print("\n Files saved:")
print(" - synthetic_patient_data_with_anomalies.xlsx")
print(" - synthetic_patient_data_cleaned.xlsx")
