from DataPrepossessing import f_data
from sklearn.model_selection import train_test_split
from decisionTree import DecisionTree
from RandomForest import RandomForest
from test import featureExtraction

import numpy as np
import pandas as pd

y = f_data['Label']
x = f_data.drop('Label', axis=1)


print("start\n")

    
# print(y.head())
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)
#print(y_train[2])



# clf = DecisionTree()
# clf.fit(X_train, y_train)
# predections = clf.predict(X_test)

# def accuracy(y_test, y_pred):
#     return np.sum(y_test == y_pred) / len(y_test)


# acc = accuracy(y_test, predections)
# print("ACCURACY=",acc*100)




def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy



#------------------------------------------------------------------
clf = RandomForest(n_trees=75)
clf.fit(X_train, y_train)
# print(type(X_test))
# dataq = pd.read_csv("Book2.csv")
# dataqq=dataq.drop(['Domain'],axis=1)

predictions = clf.predict(X_train)
# print(predictions)


acc =  accuracy(y_train, predictions)
print(acc)

class checkURL():
    def __init__(self):
        data=[]
        url ="https://www.google.com/search?q=school&sxsrf=AJOqlzUs5VKSA_ViZCS55vYbsSf018qRPQ%3A1674465781665&ei=9VHOY5qZKJuTseMP4r--iAM&ved=0ahUKEwja5-qQr938AhWbSWwGHeKfDzEQ4dUDCA8&uact=5&oq=school&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIICAAQsQMQkQIyBQgAEJECMgQIABBDMgQIABBDMgQIABBDMggIABCABBCxAzIICAAQgAQQsQMyCAgAEIAEELEDMgsIABCABBCxAxCDATIICC4QgAQQsQM6BAgjECc6BAguEEM6CAguELEDEIMBOgsILhCDARCxAxCABDoICAAQsQMQgwE6BwguENQCEEM6CggAELEDEIMBEEM6CgguELEDENQCEEM6BQgAEIAEOgsILhCABBCxAxCDAUoECEEYAEoECEYYAFAAWNwMYIEPaABwAHgAgAHlAYgB_QaSAQUwLjUuMZgBAKABAcABAQ&sclient=gws-wiz-serp"


        data.append(featureExtraction(url))
        print("data=",data)

        # # # Convert the array to a DataFrame
        
        df = pd.DataFrame(data,columns=['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards'])
        print(df)
        print(df.shape)
        predictions = clf.predict(df)
        print("pre=",predictions)

c= checkURL()

from joblib import dump

dump(clf, './model.joblib')