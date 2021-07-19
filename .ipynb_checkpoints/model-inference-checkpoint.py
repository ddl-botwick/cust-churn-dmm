import pandas as pd
import numpy as np
import random
import math
import pickle
import json
import os
import requests
import upload_to_s3

#July 19 comment
#Read in training data
print('Reading in training data')
df = pd.read_csv('/mnt/data/smallPrepared.csv')

#Double volume of data
df2 = df.append(df)

#Reset custid field so that there are no repeats
df2['custid'] = np.random.choice(range(df.custid.min(), df.custid.max()),df2.shape[0], replace=False)

#### Choose a random percentage between a set range to scale a variable by
# Cast all integer vars back to integers
droppJitter = df2.dropperc.apply(lambda x : x*(random.randrange(50,150))/100)
minsJitter = df2.mins.apply(lambda x : x*(random.randrange(70,130)/100)).round(0).astype(int)
consecMonthsJitter = df2.consecmonths.apply(lambda x : x*(random.randrange(80,120)/100)).round(0).astype(int)
incomeJitter = df2.income.apply(lambda x : x*(random.randrange(40,160)/100)).round(1)
ageJitter = df2.age.apply(lambda x : x*(random.randrange(90,110)/100)).round(0).astype(int)

#Take all the new 'jittered' variables and write to a new df
df3 = pd.DataFrame({'custid': df2.custid,
       'dropperc': droppJitter, 
       'mins': minsJitter,
       'consecmonths': consecMonthsJitter,
       'income': incomeJitter,
       'age': ageJitter,
       'churn_Y': df2.churn_Y
                   })

#Grab between 100 and 300 records from new jittered data
df_inf = df3.sample(n = random.randint(100,300)).reset_index(drop = True)

#Load in previously trained classification model
loaded_model = pickle.load(open('/mnt/Models/BestModelCV.pkl', 'rb'))

#Take input features and score them against loaded in model
print('Making predictions against new data')
X = df_inf.loc[:, 'dropperc':'age']
predictions = loaded_model.predict(X)

#Create dataframe with just customer id and ground truth churn colum
groundTruth = pd.DataFrame(df_inf[['custid','churn_Y']]).rename(columns = {'churn_Y': 'y_gt'})

#Take array of predictions and create a df with index from gt data
preds_df = pd.DataFrame(data=predictions, columns=['churn_Y'], index=groundTruth.index)

#Add in input features and custid to above df for final input and preds data
prod_inputs_and_preds = df_inf.drop('churn_Y', axis =1).join(preds_df)


#prediction and ground truth file export
print('Saving ground truth and prediction data')
prod_pred_partitions = upload_to_s3.split_data_export(prod_inputs_and_preds,1,'prod_inputs_and_preds')        
ground_truth_partitions = upload_to_s3.split_data_export(groundTruth,1,'prod_ground_truth')
 
#save latest file names
file_name_dict = {}
file_name_dict['prod_predictions'] = prod_pred_partitions
file_name_dict['prod_ground_truth'] = ground_truth_partitions
 
#Update latest files json 
file_names = json.dumps(file_name_dict)
f = open("/mnt/temp/prod_files_latest.json","w")
f.write(file_names)
f.close()

#Specify s3 bucket for uploading
bucket = 'dmm-eb'

#Upload to S3 bucket
print('Writing to S3 bucket')
for name in prod_pred_partitions:
    upload_to_s3.upload(name, bucket)
    
for name in ground_truth_partitions:
    upload_to_s3.upload(name, bucket)
    
#register with DMM model
 
with open('/mnt/temp/active_model_version.json') as json_file:
    model_version_dict = json.load(json_file)
    json_file.close()
    
model_version = model_version_dict['model_version']
 
with open('/mnt/temp/prod_files_latest.json') as json_file: 
    data = json.load(json_file)
    
preds = data['prod_predictions']
ground_truth = data['prod_ground_truth']
                    
#Add prediction data to DMM
print('Adding prediction data to DMM')
for name in preds:
    
    file_name = os.path.basename(name)
    
    #Customer churn "prod"
    
    url = "https://trial.dmm.domino.tech/api/v0/models/"+model_version+"/add_predictions"
 
 
    payload = "{\n  \"dataLocation\": \"https://s3.us-east-2.amazonaws.com/"+bucket+"/"+file_name+"\"\n}"
    headers = {
               'Authorization': 'eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVlZThjMDA0YjJjYjJmNTliMGRiYmU4NiIsInVzZXJuYW1lIjoiY29saW4uZGVtby5kZGwudHJpYWwiLCJ1c2VyX3R5cGUiOiJhcGkiLCJvcmdhbml6YXRpb25faWQiOiI1ZWQ2NWRhZGQ5ZmFiZjAwMDE0ZjI5MmYifQ.bTuQUr57LtU9By6pXUP_TVMCj_MIbhYLo4ULamcafWJogx4oe8r_p8tQ5xARFQiJQzoOQL2u9-GO5FS6y7Wgpw',
               'Content-Type': 'application/json'
              }
 
    response = requests.request("PUT", url, headers=headers, data = payload)
 
    print(response.text.encode('utf8'))
                    
#Finally add ground truth data                    
print('Adding ground truth data to DMM for analysis')
                    
for name in ground_truth:
    
    file_name = os.path.basename(name)
    
    url = "https://trial.dmm.domino.tech/api/v0/models/"+model_version+"/add_ground_truths"
 
    payload = "{\n  \"dataLocation\": \"https://s3.us-east-2.amazonaws.com/"+bucket+"/"+file_name+"\"\n}"
    headers = {'Authorization': 'eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVlZThjMDA0YjJjYjJmNTliMGRiYmU4NiIsInVzZXJuYW1lIjoiY29saW4uZGVtby5kZGwudHJpYWwiLCJ1c2VyX3R5cGUiOiJhcGkiLCJvcmdhbml6YXRpb25faWQiOiI1ZWQ2NWRhZGQ5ZmFiZjAwMDE0ZjI5MmYifQ.bTuQUr57LtU9By6pXUP_TVMCj_MIbhYLo4ULamcafWJogx4oe8r_p8tQ5xARFQiJQzoOQL2u9-GO5FS6y7Wgpw',
               'Content-Type': 'application/json'
              }
 
    response = requests.request("PUT", url, headers=headers, data = payload)
 
    print(response.text.encode('utf8'))
                    
                    
                    
print('Model inference and data uploading complete - see updated dashboard at https://trial.dmm.domino.tech/model-dashboard')                    
