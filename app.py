import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu

model = pickle.load(open('rfc.pkl','rb'))

# Add a title and description
st.title('Breast Cancer Diagnosis Web Application')

# Add option menu to sidebar
with st.sidebar:
    choose = option_menu("Main Menu", ["About", "Breast Cancer Diagnosis", "Contact"],
                         icons=['house', 'kanban', 'person lines fill'],
                         menu_icon="app-indicator", default_index=0)

if choose == "About":
    st.write(':dart: This is the Web Application for Breast cancer diagnosis using machine learning based on parameters collected from physical examinations and routine blood analysis.')
    st.write(':thermometer: Clinical features, including Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin and MCP-1 will be used to predict breast cancer.')

elif choose == "Breast Cancer Diagnosis":
    st.markdown('## Breast Cancer related features')
    st.markdown('### Physical features')
    Age = st.slider('Age (years)', 0.0, 99.0, 1.0)
    BMI = st.slider('BMI (kg/m2)', 15.0, 40.0, 0.1)
    st.markdown('### Blood features')
    Glucose = st.slider('Glucose (mg/dL)', 1.0, 300.0, 0.1)
    Insulin = st.slider('Insulin (µU/mL)', 0.0, 100.0, 0.1)
    HOMA = st.slider('HOMA', 0.0, 50.0, 0.1)
    Leptin = st.slider('Leptin (ng/mL)', 0.0, 120.0, 0.1)
    Adiponectin = st.slider('Adiponectin (µg/mL)', 0.0, 50.0, 0.1)
    Resistin = st.slider('Resistin (ng/mL)', 0.0, 120.0, 0.1)
    MCP1 = st.slider('MCP-1 (pg/dL)', 20.0, 2000.0, 1.0)
    
    if st.button('Breast Cancer Diagnosis'):
        prediction = model.predict([[Age, BMI, np.log(Glucose), np.log(Insulin), np.log(HOMA), np.log(Leptin), np.log(Adiponectin), np.log(Resistin), np.log(MCP1)]])
        if prediction == 1:
            st.balloons()
            st.success('You are not breast cancer patients')
        else:
             st.success('You are breast cancer patients')
       
elif choose == "Contact":
    st.write(':face_with_cowboy_hat: Author: Wunchana Seubwai')
    st.write(':email: Email: wseubwai@iu.edu')

