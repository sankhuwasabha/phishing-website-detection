from django.shortcuts import render
import numpy as np
from test import featureExtraction
import pandas as pd
from joblib import load
model = load('./model.joblib')

# Create your views here.

def predict(request):
    if request.method=="POST":
        input_url=request.POST['url1']
        features=[]
        features.append(featureExtraction(input_url))
        df = pd.DataFrame(features,columns=['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards'])
        detect_url=model.predict(df)

        if detect_url==0:
            detect_url = f'Letimate URL{detect_url}'
        else:
            detect_url = f'Phishing URL{detect_url}'
        return render(request,'index.html',{'output': detect_url})
    return render(request, 'index.html')
