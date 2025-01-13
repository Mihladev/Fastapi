import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd


app = FastAPI()


openai.api_key = "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"


@app.on_event("startup")
def load_data():
    global entry_level, mid_level, senior_level, all_data
    try:
        entry_level = pd.read_csv("C:/Users/Student/Desktop/JMA/BE/data/entry_level.csv")
        mid_level = pd.read_csv("C:/Users/Student/Desktop/JMA/BE/data/mid_level.csv")
        senior_level = pd.read_csv("C:/Users/Student/Desktop/JMA/BE/data/senior_level.csv")
        all_data = pd.read_csv("C:/Users/Student/Desktop/JMA/BE/data/Team_1.csv")  
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Dataset files not found. Ensure 'entry_level.csv', 'mid_level.csv', 'senior_level.csv' are in the 'data/' folder.")


class FilterRequest(BaseModel):
    experience_level: str

class QueryRequest(BaseModel):
    query: str

# AI Query Function 
def get_ai_insights(query: str):
    """Function to send query to OpenAI and retrieve response."""
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  
            prompt=f"Analyze the following job market data and answer the question: {query}",
            max_tokens=200,
            temperature=0.5,
        )
        answer = response.choices[0].text.strip()
        return answer
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with AI request: {e}")

# API Endpoints
@app.get("/data/preview")
def preview_data():
    """Get a preview of the dataset."""
    return entry_level.head(10).to_dict(orient="records")

@app.post("/data/filter")
def filter_data(filter_request: FilterRequest):
    """Filter data based on experience level."""
    experience_level = filter_request.experience_level.lower()
    
    if experience_level == "all":
        filtered_jobs = all_data
    elif experience_level == "entry-level":
        filtered_jobs = entry_level
    elif experience_level == "mid-level":
        filtered_jobs = mid_level
    elif experience_level == "senior-level":
        filtered_jobs = senior_level
    else:
        raise HTTPException(status_code=400, detail="Invalid experience level.")
    
    if filtered_jobs.empty:
        return {"message": f"No jobs found for the selected experience level: {filter_request.experience_level}."}

    return filtered_jobs.to_dict(orient="records")

@app.post("/ai/query")
def ai_query(query_request: QueryRequest):
    """Handle user queries with AI (using OpenAI)."""
    query = query_request.query
    
    # Get AI insights
    insights = get_ai_insights(query)
    return {"query": query, "insights": insights}

@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "FastAPI backend is running!"}
