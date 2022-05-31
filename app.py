from fastapi import FastAPI, File, UploadFile
import pandas as pd
from fastapi.encoders import jsonable_encoder
app = FastAPI()
from io import BytesIO

@app.post("/upload-file/")
async def create_upload_file(csv_file: UploadFile = File(...)):
    print(csv_file.filename)
    dataframe = pd.read_csv(BytesIO(csv_file.file.read()))
    Output_JSON = dataframe.to_json(orient='table')

    return Output_JSON
