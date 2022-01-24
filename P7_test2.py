import uvicorn
import pickle
import pandas as pd
from fastapi import FastAPI

from pydantic import BaseModel

app = FastAPI()

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
X_test= pd.read_pickle("test_data.pkl")
print(type(X_test))

class creditnote(BaseModel):
    index:int
    
@app.get('/{name}')
async def get_name(name: str):
    '''
    This is a test : My first API
    '''
    return {'message': f'Hello, {name}'}

@app.post('/predict')
async def predict_creditnote(data_api: creditnote):
    data_api = data_api.dict()
    print("avant")
    index=int(data_api['index'])
    print("après")
    
    prediction = loaded_model.predict_proba(X_test.iloc[[index]])

    
    if(prediction[0][0]>0.5):
        prediction="solvent customer"
        probability=prediction[0][0]
    else:
        prediction:"bankrupt customer"
        probability=prediction[0][1]
    return {
'prediction': prediction,'probability'=probability
       
}
