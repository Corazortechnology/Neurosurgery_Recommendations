from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_handler import QdrantStore
from typing import List, Optional
from nlp_services.sentiment_analysis import SentimeAnalysis
from nlp_services.emotions_analysis import EmotionsAnalysis
from nlp_services.behaviour_analysis import BehaviourAnalysis
from nlp_services.summarize import Summarizer
from recommendation import Recommendation
import logging
import json
logging.basicConfig(level=logging.INFO)
app = FastAPI()

class QueryRequest(BaseModel):
    user_track_journey: dict
    user_journey : dict
    user_name:str
    user_age:int
    user_id:str
    k: Optional[int] = 2

class ChatRequest(BaseModel):
    query :str
    user_id:str
    k: Optional[int] = 2

class UpdateRequest(BaseModel):
    id: int
    new_text: str
    new_metadata: Optional[dict] = None

recommender = Recommendation()

@app.post("/update_text")
async def update_text(request: UpdateRequest):
    qdrant_handler = QdrantStore(collection_name="neurosurgery", url="http://localhost:6333")
    qdrant_handler.update_text(request.id, request.new_text, request.new_metadata)
    return {"message": "Text updated successfully"}

@app.post("/recommedation")
async def get_recommendation(request: QueryRequest):
    user_track_journey = request.user_track_journey
    user_journey = request.user_journey
    user_name = request.user_name
    user_age = request.user_age
    query = {**user_track_journey, **user_journey, "user_name": user_name, "user_age": user_age}
    logging.info(f"Payload: {query}")
    user_id = request.user_id
    qdrant_handler = QdrantStore(collection_name="neurosurgery", url="http://localhost:6333")
    sentiment_analyzer = SentimeAnalysis()
    emotion_analyzer = EmotionsAnalysis()
    behaviour_analyzer = BehaviourAnalysis()
    
    # behaviour_analysis = behaviour_analyzer.analyze_gemini(query)
    behaviour_analysis = behaviour_analyzer.analyze(query,"llama-3.3-70b-versatile")
    behaviour_analysis = json.loads(behaviour_analysis)
    logging.info(f"Behaviour analysis: {behaviour_analysis}")
    summarizer = Summarizer()
    profile_summary = summarizer.analyze(query,"llama-3.3-70b-versatile")
    logging.info(f"Profile Summary: {profile_summary}")
    sentiment = sentiment_analyzer.sentiment_analyze(behaviour_analysis['summary'])
    logging.info(f"Sentiment analysis: {sentiment}")
    emotion = emotion_analyzer.emotion_analysis(behaviour_analysis['summary'])
    logging.info(f"Emotional analysis: {emotion}")
    results = qdrant_handler.similarity_search(behaviour_analysis['summary'], k=2)

    recommendations = recommender.recommend(
        user_id,
        context_vars={
            "patient_profile": query,
            "retrieved_text": results[0].page_content if results else "",
            "sentiment_analysis": sentiment,
            "emotional_state": emotion,
            "behavioral_analysis": behaviour_analysis
        }
    )
    logging.info(f"Recommendation: {recommendations}")
    
    return {"recommendations": recommendations}

@app.post("/chat")
async def get_recommendation(request: ChatRequest):
    query = request.query
    user_id = request.user_id
    qdrant_handler = QdrantStore(collection_name="neurosurgery", url="http://localhost:6333")
    sentiment_analyzer = SentimeAnalysis()
    emotion_analyzer = EmotionsAnalysis()
    behaviour_analyzer = BehaviourAnalysis()
    
    # behaviour_analysis = behaviour_analyzer.analyze_gemini(query)
    behaviour_analysis = behaviour_analyzer.analyze(query,"llama-3.3-70b-versatile")
    behaviour_analysis = json.loads(behaviour_analysis)
    logging.info(f"Behaviour analysis: {behaviour_analysis}")
    summarizer = Summarizer()
    profile_summary = summarizer.analyze(query,"llama-3.3-70b-versatile")
    logging.info(f"Profile Summary: {profile_summary}")
    sentiment = sentiment_analyzer.analyze(query)
    logging.info(f"Sentiment analysis: {sentiment}")
    emotion = emotion_analyzer.analyze(query)
    logging.info(f"Emotional analysis: {emotion}")
    results = qdrant_handler.similarity_search(query, k=2)

    recommendations = recommender.recommend(
        user_id,
        context_vars={
            "patient_profile": query,
            "retrieved_text": results[0].page_content if results else "",
            "sentiment_analysis": sentiment,
            "emotional_state": emotion,
            "behavioral_analysis": behaviour_analysis
        }
    )
    logging.info(f"Recommendation: {recommendations}")
    
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
