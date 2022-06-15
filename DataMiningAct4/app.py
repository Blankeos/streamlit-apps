import streamlit as st
import pandas as pd
from keras.models import load_model

st.write("""
# 🧠 Gender Population Suicide Rate Regression using Neural Network Model
by Glecy Elizalde, Christopher Joseph Rubinos, Carlo Antonio T. Taleon (BSCS-3A)
""")

st.sidebar.header('💻 User Input Parameters')

model = load_model("DataMiningAct4\gender_suicide_rate_regression_model.h5")


def user_input_features():
    gender = st.sidebar.radio(f'⚥ Gender', ('Male', 'Female'))
    education_background = st.sidebar.radio(
        '🏫 Education Background', ('Advanced', 'Basic', 'Intermediate'))
    unemployment = st.sidebar.slider(
        '😿 Unemployment Rate %', min_value=0.0, max_value=1.0, value=0.44)
    credit_card_ownership = st.sidebar.slider(
        '💳 Credit Card Ownership %', min_value=0.0, max_value=1.0, value=0.44)
    debit_card_ownership = st.sidebar.slider(
        "💳 Debit Card Ownership %", min_value=0.0, max_value=1.0, value=0.44)
    data = {'gender': gender,
            'education_background': education_background,
            'unemployment': unemployment,
            'credit_card_ownership': credit_card_ownership,
            'debit_card_ownership': debit_card_ownership}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.markdown(f"""
    ### Your Parameters:
    - **⚥ Gender:** {'👦 Male' if df.iloc[0]['gender']=="Male" else '👧 Female'}
    - **🏫 Education Background:** {df.iloc[0]['education_background']}
    - **😿 Unemployment Rate:** {df.iloc[0]['unemployment']*100}%
    - **💳 Credit Card Ownership:** {df.iloc[0]['credit_card_ownership']*100}%
    - **💳 Debit Card Ownership:** {df.iloc[0]['debit_card_ownership']*100}%
""")


def transform_df(df):
    gender = {'Female': 0, 'Male': 1, }
    education_background = {"Advanced": 0, "Basic": 1, "Intermediate": 2}
    df['gender'] = gender[df.iloc[0]['gender']]
    df['education_background'] = education_background[df.iloc[0]
                                                      ['education_background']]
    return df


st.markdown("""
    ### Transformed Dataframe:
""")
df = transform_df(df)
st.dataframe(df)


pred = model.predict(df)
st.markdown(f"""
    ### 🤖 Model Output:
    - **Predicted Suicide Rate:** {pred[0][0]*100}%
""")
