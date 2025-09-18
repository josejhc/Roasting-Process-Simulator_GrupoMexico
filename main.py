import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
# from functions.project import project, load_columns 
from functions.proXGB import proXGB, load_columns

# Initialize the FastAPI application
app = FastAPI(
    title="Prediction API",
    description="API for making predictions using a custom model.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model for prediction requests
class PredictionRequest(BaseModel):
    inputs: List[float]
    targetVariable: str
    weight: float 

# Output model for prediction responses (optional but recommended)
class PredictionResponse(BaseModel):
    received: List[float]
    results: List[float]
    stats: List[Dict[str, Any]]
    final_pressure: float

     

# Prediction endpoint
@app.post("/request", response_model=PredictionResponse)
async def prediction(data: PredictionRequest):
    # Convert input list to NumPy array and reshape for model input
    input_vector = np.array(data.inputs).reshape(1, -1)
    target = data.targetVariable
    weight = data.weight

    # Call the prediction function
    # result, stats, X_columns, Y_columns = project(input_vector, target)
    result, stats, stored_inputs = proXGB(input_vector, target,weight)

    flat_result = np.array(result).flatten().tolist()
    # presion = stored_inputs[11]
    # Return structured response
    return PredictionResponse(
        received=data.inputs,
        results=flat_result,
        stats = stats,
        final_pressure=stored_inputs[11]        
    )

#Get columns
class ColumnsResponse(BaseModel):
    Y_columns: List[str]
    X_columns: List[str]

@app.get("/columns", response_model=ColumnsResponse)
async def get_column_names():
    X_columns, Y_columns = load_columns()
    return ColumnsResponse(
        X_columns=X_columns,
        Y_columns=Y_columns
    )



#If you wan to change a column (add or remove) you've to modify project.py (both functions)