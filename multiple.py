import pickle
from matplotlib.ft2font import HORIZONTAL
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

diabetes_model = pickle.load(open("C:/Users/sahas/OneDrive/Documents/visual studio code/doctor/diabetes_model.sav","rb"))
heart_model = pickle.load(open("C:/Users/sahas/OneDrive/Documents/visual studio code/doctor/heart_disease_model.sav","rb"))
parkinson_model = pickle.load(open("C:/Users/sahas/OneDrive/Documents/visual studio code/doctor/parkinsons_model.sav","rb"))




with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System", ['Doctor Fee Price','Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction','Drug Recommendation'],
                         icons=['currency-dollar','activity','heart','person',''],
                         menu_icon="app-indicator", 
                         default_index=0,orientation=HORIZONTAL,
                        
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
if (selected=='Doctor Fee Price'):
    st.title('Fee Price Prediction')
    
    model = pickle.load(open(r'C:/Users/sahas/OneDrive/Documents/visual studio code/doctor/doctor.pkl','rb'))
    #CGPA
    Profile = st.selectbox(
     'Select Doctor Profile',
     (('Ayurveda','Dentist','Dermatologists','ENT_Specialist','General_Medicine','Homeopath'))
     )
    if(Profile == "Ayurveda"):
        Profile_Ayurveda = 1
        Profile_Dentist = 0
        Profile_Dermatologists = 0
        Profile_ENT_Specialist = 0
        Profile_General_Medicine = 0
        Profile_Homeopath = 0
        
    elif(Profile == "Dentist"):
            Profile_Ayurveda = 0
            Profile_Dentist = 1
            Profile_Dermatologists = 0
            Profile_ENT_Specialist = 0
            Profile_General_Medicine = 0
            Profile_Homeopath = 0
            
    elif(Profile == "Dermatologists"):
            Profile_Ayurveda = 0
            Profile_Dentist = 0
            Profile_Dermatologists = 1
            Profile_ENT_Specialist = 0
            Profile_General_Medicine = 0
            Profile_Homeopath = 0
            
    elif(Profile == "ENT Specialist"):
            Profile_Ayurveda = 0
            Profile_Dentist = 0
            Profile_Dermatologists = 0
            Profile_ENT_Specialist = 1
            Profile_General_Medicine = 0
            Profile_Homeopath = 0
            
    elif(Profile == "General Medicine"):
            Profile_Ayurveda = 0
            Profile_Dentist = 0
            Profile_Dermatologists = 0
            Profile_ENT_Specialist = 0
            Profile_General_Medicine = 1
            Profile_Homeopath = 0
            
    elif(Profile == "Homeopath"):
            Profile_Ayurveda = 0
            Profile_Dentist = 0
            Profile_Dermatologists = 0
            Profile_ENT_Specialist = 0
            Profile_General_Medicine = 0
            Profile_Homeopath = 1
            
    else:
            Profile_Ayurveda = 0
            Profile_Dentist = 0
            Profile_Dermatologists = 0
            Profile_ENT_Specialist = 0
            Profile_General_Medicine = 0
            Profile_Homeopath = 0

    #GRE Score
    Qualification = st.selectbox(
     'Select Qualification of Doctor',
     ('MBBS','BDS','BAMS','BHMS','MD - Dermatology','MS - ENT','Venereology & Leprosy','MD - General Medicine','Diploma in Otorhinolaryngology (DLO)','MD')
     )
    if (Qualification == "MBBS"):
            MBBS = 1
            BDS = 0
            BAMS = 0
            BHMS = 0
            MD_Dermatology = 0
            MS_ENT = 0
            Venereology_Leprosy = 0
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 0
            
    elif (Qualification == "BDS"):
            MBBS = 0
            BDS = 1
            BAMS = 0
            BHMS = 0
            MD_Dermatology = 0
            MS_ENT = 0
            Venereology_Leprosy = 0
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 0
            
    elif (Qualification == "BAMS"):
            MBBS = 0
            BDS = 0
            BAMS = 1
            BHMS = 0
            MD_Dermatology = 0
            MS_ENT = 0
            Venereology_Leprosy = 0
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 0
            
    elif (Qualification == "BHMS"):
            MBBS = 0
            BDS = 0
            BAMS = 0
            BHMS = 1
            MD_Dermatology = 0
            MS_ENT = 0
            Venereology_Leprosy = 0
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 0
            
    elif (Qualification == "MD Dermatology"):
            MBBS = 0
            BDS = 0
            BAMS = 0
            BHMS = 0
            MD_Dermatology = 1
            MS_ENT = 0
            Venereology_Leprosy = 0
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 0
            
    elif (Qualification == "MS ENT"):
            MBBS = 0
            BDS = 0
            BAMS = 0
            BHMS = 0
            MD_Dermatology = 0
            MS_ENT = 1
            Venereology_Leprosy = 0
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 0
            
    elif (Qualification == "Venereology and Leprosy"):
            MBBS = 0
            BDS = 0
            BAMS = 0
            BHMS = 0
            MD_Dermatology = 0
            MS_ENT = 0
            Venereology_Leprosy = 1
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 0
            
    elif (Qualification == "MD General Medicine"):
            MBBS = 0
            BDS = 0
            BAMS = 0
            BHMS = 0
            MD_Dermatology = 0
            MS_ENT = 0
            Venereology_Leprosy = 0
            MD_General_Medicine = 1
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 0
            
    elif (Qualification == "Diploma in Otorhinolaryngology"):
            MBBS = 0
            BDS = 0
            BAMS = 0
            BHMS = 0
            MD_Dermatology = 0
            MS_ENT = 0
            Venereology_Leprosy = 0
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 1
            MD_Homeopathy = 0
            
    elif (Qualification == "MD Homeopathy"):
            MBBS = 0
            BDS = 0
            BAMS = 0
            BHMS = 0
            MD_Dermatology = 0
            MS_ENT = 0
            Venereology_Leprosy = 0
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 1
            
    else:
            MBBS = 0
            BDS = 0
            BAMS = 0
            BHMS = 0
            MD_Dermatology = 0
            MS_ENT = 0
            Venereology_Leprosy = 0
            MD_General_Medicine = 0
            Diploma_in_Otorhinolaryngology = 0
            MD_Homeopathy = 0
    st.write('You selected:', Qualification)
    #TOEFL Score
    Experience = st.slider("Enter years of Experience", min_value=0, max_value=40,  step=1)
    #university_rating
    Rating = st.slider("Rating of the Doctor", min_value=1, max_value=5,  step=1)
    #Letter of Recommendation
    City = st.selectbox(
     'Select City',
     (' Bangalore', ' Mumbai', ' Delhi', ' Hyderabad', ' Chennai', ' Coimbatore', ' Ernakulam', ' Thiruvananthapuram')
     )
    if (City == "Bangalore"):
            city_Bangalore = 1
            city_Chennai = 0
            city_Coimbatore = 0
            city_Delhi = 0
            city_Ernakulam = 0
            city_Hyderabad = 0
            city_Mumbai = 0            
            city_Thiruvananthapuram = 0
            city_Unknown = 0
            
    elif (City == "Chennai"):
            city_Bangalore = 0
            city_Chennai = 1
            city_Coimbatore = 0
            city_Delhi = 0
            city_Ernakulam = 0
            city_Hyderabad = 0
            city_Mumbai = 0            
            city_Thiruvananthapuram = 0
            city_Unknown = 0
            
    elif (City == "Coimbatore"):
            city_Bangalore = 0
            city_Chennai = 0
            city_Coimbatore = 1
            city_Delhi = 0
            city_Ernakulam = 0
            city_Hyderabad = 0
            city_Mumbai = 0            
            city_Thiruvananthapuram = 0
            city_Unknown = 0
            
    elif (City == "Delhi"):
            city_Bangalore = 0
            city_Chennai = 0
            city_Coimbatore = 0
            city_Delhi = 1
            city_Ernakulam = 0
            city_Hyderabad = 0
            city_Mumbai = 0            
            city_Thiruvananthapuram = 0
            city_Unknown = 0
            
    elif (City == "Ernakulam"):
            city_Bangalore = 0
            city_Chennai = 0
            city_Coimbatore = 0
            city_Delhi = 0
            city_Ernakulam = 1
            city_Hyderabad = 0
            city_Mumbai = 0            
            city_Thiruvananthapuram = 0
            city_Unknown = 0
            
    elif (City == "Hyderabad"):
            city_Bangalore = 0
            city_Chennai = 0
            city_Coimbatore = 0
            city_Delhi = 0
            city_Ernakulam = 0
            city_Hyderabad = 1
            city_Mumbai = 0            
            city_Thiruvananthapuram = 0
            city_Unknown = 0
            
    elif (City == "Mumbai"):
            city_Bangalore = 0
            city_Chennai = 0
            city_Coimbatore = 0
            city_Delhi = 0
            city_Ernakulam = 0
            city_Hyderabad = 0
            city_Mumbai = 1            
            city_Thiruvananthapuram = 0
            city_Unknown = 0
            
    elif (City == "Thiruvananthapuram"):
            city_Bangalore = 0
            city_Chennai = 0
            city_Coimbatore = 0
            city_Delhi = 0
            city_Ernakulam = 0
            city_Hyderabad = 0
            city_Mumbai = 0            
            city_Thiruvananthapuram = 1
            city_Unknown = 0
            
    else:
            city_Bangalore = 0
            city_Chennai = 0
            city_Coimbatore = 0
            city_Delhi = 0
            city_Ernakulam = 0
            city_Hyderabad = 0
            city_Mumbai = 0            
            city_Thiruvananthapuram = 0
            city_Unknown = 1

    if st.button('Predict Probability'):
        prediction = model.predict([[Experience,Rating,MBBS,BDS,BAMS,BHMS,MD_Dermatology,MS_ENT,Venereology_Leprosy,
                MD_General_Medicine,Diploma_in_Otorhinolaryngology,MD_Homeopathy,city_Bangalore,city_Chennai,city_Coimbatore,
                city_Delhi,city_Ernakulam,city_Hyderabad,city_Mumbai,city_Thiruvananthapuram,Profile_Ayurveda,Profile_Dentist,
                Profile_Dermatologists,Profile_ENT_Specialist,Profile_General_Medicine,Profile_Homeopath
            ]])
        Fees = round(prediction[0],2) 
        updated_res = Fees.flatten().astype(float) * 3
        st.success('The Doctor will charge you INR {}'.format(updated_res))

if (selected=='Diabetes Prediction'):
    st.title("Diabetes Prediction using ML")

    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure value")
    with col1:
        SkinThickness = st.text_input("Skin Thickness value")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    with col2:
        Age = st.text_input("Age of the person")

    diab_diagnosis=" "
    if st.button("Diabetes Test Result"):
        diab_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
     
        if (diab_prediction[0]==1):
            diab_diagnosis = "The person is Diabetic"
        else:
            diab_diagnosis = "The person is not Diabetic"
    
    st.success(diab_diagnosis)

if(selected == "Heart Disease Prediction"):
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')        
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')      
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')    
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)

if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinson_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

if (selected=='Drug Recommendation'):
        
    url = 'https://drive.google.com/file/d/1-0FlARiLI7RMiKCUYwAOGBCXGI1895j4/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]

    data = pd.read_csv(path)

    st.title("Drug Recommendation System")

    popular_conditions = ('Birth Control','Depression','Pain','Anxiety','Acne','Bipolar Disorde','Insomnia','Weight Loss',
                        'Obesity','ADHD', 'Diabetes, Type 2', 'Emergency Contraception', 'High Blood Pressure','Migrane')
    conditions = data.loc[data['condition'].isin(popular_conditions)]


    user_choice = st.selectbox("Select Condition", popular_conditions)

    col1, col2 = st.columns(2)

    with col1:
        st.title("Top 5 Drugs")
        st.dataframe(data[data['condition'] == user_choice][['drugName','usefulness']].sort_values(by = 'usefulness',
                                                    ascending = False).head().reset_index(drop = True))
    with col2:
        st.title("Bottom 5 Drugs")
        st.dataframe(data[data['condition'] == user_choice][['drugName','usefulness']].sort_values(by = 'usefulness',
                                                    ascending = True).head().reset_index(drop = True))