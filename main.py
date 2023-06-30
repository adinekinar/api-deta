from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class inputRekom (BaseModel):
    nik: int
    jk: float
    bbl: float
    pbl: float
    bb: float
    pb: float
    umur: int
    bbu: float
    zsbbu: float
    pbu: float
    zspbu: float
    bbpb: float
    zsbbpb: float

with open('randomforest.pkl','rb') as f:
    model = pickle.load(f)

@app.post('/')
async def recom_endpoint(item: inputRekom):
    pred = model.predict(pd.DataFrame([[item.nik, item.jk, item.bbl, item.pbl, item.bb, item.pb, item.umur, item.bbu, item.zsbbu, item.pbu, item.zspbu, item.bbpb, item.zsbbpb]],
                        columns=['NIK', 'JK', 'BB Lahir', 'TB Lahir', 'BB', 'TB', 'Umur', 'BB/U', 'ZS BB/U', 'TB/U', 'ZS TB/U', 'BB/TB', 'ZS BB/TB']))

    return int(pred)