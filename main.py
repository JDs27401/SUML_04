import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model(path: str):
    with open(path, 'rb') as f:
        m = pickle.load(f)
        return m

def predict_from_model(model, Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked):
    if Embarked == 'Queenstown':
        Embarked_Q = 1
        Embarked_S = 0
    else:
        Embarked_Q = 0
        Embarked_S = 1

    data = pd.DataFrame([{
        'Pclass': Pclass,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Sex_male': Sex_male,
        'Embarked_Q': Embarked_Q,
        'Embarked_S': Embarked_S
    }])

    prediction = model.predict(data)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(data)[0][1]
    else:
        proba = None

    return prediction, proba

def main():
    st.set_page_config(page_title="Titanic Predictor", layout="centered")

    bg_url = "https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🚢 Titanic Survival Predictor")

    model = load_model('titanic_model.pkl')

    Pclass = st.selectbox("Klasa pasażera", [1, 2, 3])

    Age = st.number_input("Wiek", min_value=1, max_value=99, value=25)

    SibSp = st.number_input("Liczba rodzeństwa / małżonków", min_value=0, max_value=10, value=0)

    Parch = st.number_input("Liczba rodziców / dzieci", min_value=0, max_value=10, value=0)

    Fare = st.number_input("Cena biletu", min_value=0.0, max_value=1000.0, value=50.0)

    Sex = st.radio("Płeć", ["Kobieta", "Mężczyzna"])
    Sex_male = 1 if Sex == "Mężczyzna" else 0

    Embarked = st.selectbox("Port zaokrętowania", ["Queenstown", "Southampton"])

    if st.button("🔮 Przewidź"):
        result, prob = predict_from_model(
            model,
            Pclass,
            Age,
            SibSp,
            Parch,
            Fare,
            Sex_male,
            Embarked
        )

        if result == 1:
            st.success("✅ Pasażer przeżyje")
        else:
            st.error("❌ Pasażer nie przeżyje")

        if prob is not None:
            st.write(f"Szansa na przeżycie: **{prob * 100:.2f}%**")

            st.progress(int(prob * 100))

if __name__ == '__main__':
    main()