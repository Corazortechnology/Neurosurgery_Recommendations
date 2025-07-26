# RAG_Chatbot

A modular FastAPI-based backend for document embedding, behavioral analysis, sentiment/emotion detection, recommendations, and report generation using LLMs and vector search.

---

## Features

- **PDF Embedding:** Store PDF content as vector embeddings in Qdrant.
- **Behavior, Sentiment, Emotion Analysis:** Analyze user text using LLMs (Groq, Gemini).
- **Personalized Recommendations:** Retrieve and recommend content based on user profile and journey.
- **Report Generation:** Generate and download user-specific PDF reports.
- **RESTful APIs:** All features are exposed as FastAPI endpoints.

---

## Requirements

- Python 3.8+
- Qdrant running locally or remotely
- API keys for Gemini, OpenAI, Groq (set in `.env`)
- [Install dependencies](#installation)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd RAG_Chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file with your API keys:
     ```
     GEMINI_API_KEY=your_gemini_key
     OPENAI_API_KEY=your_openai_key
     GROQ_API_KEY=your_groq_key
     ```

4. **Start Qdrant (if not already running):**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

---

## Running the APIs

- **Embedding API:**  
  ```bash
  python store_embedding.py
  # Runs on http://localhost:3000
  ```

- **Report Generation API:**  
  ```bash
  python report_generation_api.py
  # Runs on http://localhost:8000
  ```

- **Main Chatbot API:**  
  ```bash
  python api.py
  # Runs on http://localhost:8000
  ```

---

## API Endpoints

### 1. Store Embeddings

**POST /insert_texts**  
Store PDF embeddings in Qdrant.

**Request:**
```json
{
  "pdf_path": ["path/to/file1.pdf", "path/to/file2.pdf"]
}
```
**Query Parameters:**  
- `collection_name` (default: "neurosurgery")
- `url` (default: "http://localhost:6333")

**Response:**
```json
{ "message": "PDF texts inserted successfully" }
```

---

### 2. Report Generation

**POST /generate-report**  
Generate a PDF report for a user.

**Request:**
```json
{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "user_id": "user123"
}
```
**Response:**  
- Returns a PDF file.

---

### 3. Recommendation

**POST /recommedation**  
Get recommendations based on user journey.

**Request:**
```json
{
  "user_track_journey": {
      "Abnormal": "100%",
      "Poor Eye Contact": "100%",
      "Social Difficulty": "100%",
      "Tics and Fidgets": "100%",
      "Aggression": "100%",
      "Depression": "100%",
      "Fixations": "100%",
      "Abnormal Flat Speech": "100%",
      "Noise Sensitivity": "100%",
      "Anxiety": "100%"
    },
  "user_journey":{
    "space": "Went outside house for dinner",
    "routine": ["School Off", "Skipped TV time"],
    "people": "One Guest",
    "sleep": {
      "duration": "5hr 45m",
      "start": "10:00 PM",
      "end": "7:45 AM",
      "graph_data": [2, 4, 1, 3, 0, 1, 3, 4, 2, 1]
    },
    "activity": [
      {
        "type": "Painting",
        "time": "13:05–15:00"
      },
      {
        "type": "Puzzles",
        "time": "17:05–18:51"
      }
    ],
    "components_changed_today": ["Anxiety", "Focus"],
    "notes": "There was a hyperactivity burst for which activity was done"
  },
  "user_name": "John Doe",
  "user_age": 30,
  "user_id": "user123",
  "k": 2
}
```
**Response:**
```json
{ "recommendations": [ ... ] }
```

---

### 4. Chat

**POST /chat**  
Get recommendations based on a free-form query.

**Request:**
```json
{
  "query": "User's query text",
  "user_id": "user123",
  "k": 2
}
```
**Response:**
```json
{ "recommendations": [ ... ] }
```

---

### 5. Update Text

**POST /update_text**  
Update a text and its metadata in Qdrant.

**Request:**
```json
{
  "id": 123,
  "new_text": "Updated text content",
  "new_metadata": { "key": "value" }
}
```
**Response:**
```json
{ "message": "Text updated successfully" }
```

---

## Notes

- All endpoints expect and return JSON unless otherwise specified.
- Make sure Qdrant and all required services are running.
- API keys must be set in your environment.

---

## License

MIT

---
