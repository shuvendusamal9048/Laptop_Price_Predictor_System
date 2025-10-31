import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('/pipe.pkl','rb'))
df = pickle.load(open('/df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

#type of laptop
type=st.selectbox('Type',df['TypeName'].unique())

#Ram
Ram=st.selectbox('RAM(in GB)',[2,4,6,8,12,16,32,64])

#weight
weight=st.number_input('Weight')

#IPS
ips=st.selectbox('IPS',['No','Yes'])

#Screen size
screen_size=st.number_input('Screen Size')

#Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    ppi=0
    # Query

    if ips=='Yes':
        ips=1
    else:
        ips=0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = np.array([company, type, Ram, weight, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 11)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))


