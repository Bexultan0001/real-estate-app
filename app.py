import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.DataFrame({
    "area": [40, 50, 60, 70, 80, 90, 100, 120],
    "floor": [1, 2, 2, 3, 3, 4, 5, 6],
    "location": ["A", "A", "B", "B", "C", "C", "A", "B"],
    "price": [15000, 20000, 25000, 30000, 35000, 40000, 50000, 65000]
})


data = pd.get_dummies(data, columns=["location"])

X = data.drop("price", axis=1)
y = data["price"]

# Train/Test бөлу
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)


st.title("🏠 Үй бағасын болжау жүйесі")

st.write("Параметрлерді енгізіңіз:")

area = st.number_input("Аудан (м²)", 10, 500, 60)
floor = st.number_input("Қабат саны", 1, 20, 2)
location = st.selectbox("Орналасу", ["A", "B", "C"])


def predict(area, floor, location):
    input_data = pd.DataFrame({
        "area": [area],
        "floor": [floor],
        "location_A": [1 if location == "A" else 0],
        "location_B": [1 if location == "B" else 0],
        "location_C": [1 if location == "C" else 0],
    })
    return model.predict(input_data)[0]


if st.button("Болжам жасау"):
    result = predict(area, floor, location)
    st.success(f"Үйдің болжамды бағасы: {int(result)} $")


if st.checkbox("Деректерді көрсету"):
    st.write(data)
