from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, PointIdsList
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from nlp_services.sentiment_analysis import SentimeAnalysis
from nlp_services.emotions_analysis import EmotionsAnalysis
from nlp_services.behaviour_analysis import BehaviourAnalysis
from recommendation import Recommendation
import json
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_query(self, text):
        return self.model.encode(text).tolist()
    def embed_documents(self, texts):
        return [self.model.encode(text).tolist() for text in texts]

class QdrantStore:
    def __init__(self, collection_name="test_collection", url="http://localhost:6333",delete=False):
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self.embeddings = SentenceTransformerEmbeddings()

        existing_collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing_collections and delete:
            self.client.delete_collection(collection_name=self.collection_name)

        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 384, "distance": "Cosine"} 
            )

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

    def insert_texts(self, texts: list, metadatas: list = None, ids: list = None):
        # Ensure each metadata has a "text" field
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for i, text in enumerate(texts):
            metadatas[i] = dict(metadatas[i])  # copy to avoid mutating input
            metadatas[i]["text"] = text
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def update_text(self, id: int, new_text: str, new_metadata: dict = None):
        vector = self.embeddings.embed_query(new_text)
        payload = dict(new_metadata) if new_metadata else {}
        payload["text"] = new_text
        point = PointStruct(
            id=id,
            vector=vector,
            payload=payload
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def delete_text(self, id: int):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[id]),
        )

    def similarity_search(self, query: str, k: int = 4):
        results = self.vectorstore.similarity_search(query, k=k)
        return results

# # Usage Example
# if __name__ == "__main__":
#     qdrant_store = QdrantStore(collection_name="neurosurgery", url="http://localhost:6333")
#     # Similarity Search
#     query = """{
#     "name": "Alex",
#     "age": 7,
#     "behaviours": ["Meltdown in classroom", "Crying", "Refused to participate"],
#     "interests": ["Music", "Drawing"],
#     "senses": {
#         "auditory": "Overwhelmed by loud noises",
#         "olfactory": None,
#         "tactile": "Uncomfortable with rough fabrics",
#         "visual": "Covered eyes due to bright lights"
#     },
#     "therapy": ["Occupational therapy"],
#     "notes": "Alex appeared very upset and anxious today. He screamed and cried when the classroom got noisy, and needed to leave the room to calm down."
# }"""es": "Alex appeared very upset and anxious today. He screamed and cried when the classroom got noisy, and needed to leave the room to calm down."
# }"""
#     behaviour_analyzer = BehaviourAnalysis()
#     sentiment_analyzer = SentimeAnalysis()
#     emotion_analyzer = EmotionsAnalysis()
#     sentiment = sentiment_analyzer.analyze(query)
#     print(f"Sentiment of user_profile: {sentiment}")
#     emotion = emotion_analyzer.analyze(query)
#     print(f"Emotion of user_profile: {emotion}")
#     behaviour_analysis = behaviour_analyzer.analyze(query)
#     behaviour_analysis = json.loads(behaviour_analysis) if behaviour_analysis else {}
#     print(f"Behaviour Analysis: {behaviour_analysis.get('label', 'No label found')}")
#     results = qdrant_store.similarity_search(query,k=2)
#     recommender = Recommendation()
#     recommendations = recommender.recommend(query, context_vars={"patient_profile": query,"retrieved_text": results[0].page_content if results else "","sentiment_analysis":sentiment,"emotional_state":emotion,"behavioral_analysis":behaviour_analysis,"feedback_data": "No previous feedback available"})
#     print(f"Recommendations: {recommendations}")