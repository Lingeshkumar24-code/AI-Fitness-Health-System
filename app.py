from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# =========================
# LOAD MODELS
# =========================
model = joblib.load("calorie_model.pkl")
model_clf = joblib.load("fitness_model.pkl")

# =========================
# LOAD & CLEAN DIET DATA
# =========================
food = pd.read_excel("nutrition.xlsx")

food.columns = food.columns.str.strip()

food = food[['name', 'calories', 'protein', 'carbohydrate', 'fat']]

food.rename(columns={
    'name': 'Food',
    'calories': 'Calories',
    'protein': 'Protein',
    'carbohydrate': 'Carbs',
    'fat': 'Fat'
}, inplace=True)

def clean_column(col):
    return col.astype(str).str.replace('g', '').str.replace('mg', '').str.replace('mcg', '').astype(float)

food['Protein'] = clean_column(food['Protein'])
food['Carbs'] = clean_column(food['Carbs'])
food['Fat'] = clean_column(food['Fat'])

food = food.dropna()

# =========================
# DIET FUNCTION
# =========================
def diet_plan(calories, goal):

    if goal == "weight_loss":
        return food[food['Calories'] <= calories * 0.5].sort_values(by='Calories').head(5)

    elif goal == "muscle_gain":
        return food[(food['Protein'] > 10) & (food['Calories'] <= calories)].sort_values(by='Protein', ascending=False).head(5)

    else:
        return food.head(5)

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    duration = float(request.form['duration'])
    heart_rate = float(request.form['heart_rate'])
    body_temp = float(request.form['body_temp'])
    gender = int(request.form['gender'])
    goal = request.form['goal']

    # INPUT DATA
    input_data = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
    columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

    # PREDICTIONS
    calories = model.predict(input_data)[0]
    fitness = model_clf.predict(input_data)[0]

    # FITNESS LABEL
    if fitness == 0:
        fitness_label = "Unfit"
    elif fitness == 1:
        fitness_label = "Moderate"
    else:
        fitness_label = "Fit"

    # DIET
    diet = diet_plan(calories, goal)

    return render_template(
        'result.html',
        calories=round(calories, 2),
        fitness=fitness_label,
        diet=diet.to_html(index=False)
    )


if __name__ == "__main__":
    app.run(debug=True)