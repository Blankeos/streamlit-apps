import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


def user_input_features():
    CRIM = st.sidebar.slider('CRIM', X.CRIM.min().tolist(
    ), X.CRIM.max().tolist(), X.CRIM.mean().tolist())
    ZN = st.sidebar.slider('ZN', X.ZN.min().tolist(),
                           X.ZN.max().tolist(), X.ZN.mean().tolist())
    INDUS = st.sidebar.slider('INDUS', X.INDUS.min().tolist(),
                              X.INDUS.max().tolist(), X.INDUS.mean().tolist())
    CHAS = st.sidebar.slider('CHAS', X.CHAS.min().tolist(
    ), X.CHAS.max().tolist(), X.CHAS.mean().tolist())
    NOX = st.sidebar.slider('NOX', X.NOX.min().tolist(),
                            X.NOX.max().tolist(), X.NOX.mean().tolist())
    RM = st.sidebar.slider('RM', X.RM.min().tolist(),
                           X.RM.max().tolist(), X.RM.mean().tolist())
    AGE = st.sidebar.slider('AGE', X.AGE.min().tolist(),
                            X.AGE.max().tolist(), X.AGE.mean().tolist())
    DIS = st.sidebar.slider('DIS', X.DIS.min().tolist(),
                            X.DIS.max().tolist(), X.DIS.mean().tolist())
    RAD = st.sidebar.slider('RAD', X.RAD.min().tolist(),
                            X.RAD.max().tolist(), X.RAD.mean().tolist())
    TAX = st.sidebar.slider('TAX', X.TAX.min().tolist(),
                            X.TAX.max().tolist(), X.TAX.mean().tolist())
    PTRATIO = st.sidebar.slider(
        'PTRATIO', X.PTRATIO.min().tolist(), X.PTRATIO.max().tolist(), X.PTRATIO.mean().tolist())
    B = st.sidebar.slider('B', X.B.min().tolist(),
                          X.B.max().tolist(), X.B.mean().tolist())
    LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min().tolist(),
                              X.LSTAT.max().tolist(), X.LSTAT.mean().tolist())
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
