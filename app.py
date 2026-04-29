import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


data = pd.DataFrame({
    "аудан": [40, 50, 60, 70, 80, 90, 100, 120],
    "қабат": [1, 2, 2, 3, 3, 4, 5, 6],
    "бөлме": [1, 2, 2, 3, 3, 4, 4, 5],
    "үй_жасы": [10, 8, 5, 7, 3, 2, 1, 1],
    "баға": [15000, 20000, 25000, 30000, 40000, 50000, 60000, 75000]
})


X = data.drop("баға", axis=1)
y = data["баға"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


st.title("🏠 Жылжымайтын мүлік бағасын болжау жүйесі")

st.write("Төмендегі мәліметтерді енгізіңіз:")

аудан = st.number_input("Аудан (м²)", 10, 1000, 60)
қабат = st.number_input("Қабат саны", 1, 20, 2)
бөлме = st.number_input("Бөлме саны", 1, 10, 2)
үй_жасы = st.number_input("Үй жасы (жыл)", 0, 100, 5)


if st.button("Бағаны болжау"):
    input_data = pd.DataFrame([[аудан, қабат, бөлме, үй_жасы]],
                              columns=["аудан","қабат","бөлме","үй_жасы"])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"💰 Болжамды баға: {int(prediction)} $")