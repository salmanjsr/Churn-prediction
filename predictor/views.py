from django.shortcuts import render
import pickle
import numpy as np
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, 'predictor', 'Perfomance_report.csv')

# CSV লোড করো
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"❌ CSV লোডে সমস্যা: {e}")

# মডেল এবং স্কেলার লোড করো
model = pickle.load(open(os.path.join(BASE_DIR, 'predictor', 'model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'predictor', 'scaler.pkl'), 'rb'))

def home(request):
    prediction = None
    if request.method == 'POST':
        try:
            age = float(request.POST.get('age'))
            work_hours = float(request.POST.get('work_hours'))
            satisfaction = float(request.POST.get('satisfaction'))
            features = np.array([[age, work_hours, satisfaction]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render(request, 'index.html', {'prediction': prediction})

