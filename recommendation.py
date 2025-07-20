from llm_service import call_gemini, call_groqapi,call_openai
from nlp_services.summarize import Summarizer
import csv
import uuid
import pandas as pd

feedback_df = pd.read_csv("/home/dell-p112f210/Documents/RAG_Chatbot/feedback.csv")

class Recommendation:
    def __init__(self, model="gemini-1.5-flash", max_output_tokens=1024, temperature=0.2,
                 rec_csv_path="recommendations.csv", feedback_csv_path="feedback.csv"):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.history = {}  # Stores user_id: [ {user_profile, recommendation, recommendation_id} ] entries
        self.response_count = {}  # Stores user_id: count
        self.rec_csv_path = rec_csv_path
        self.feedback_csv_path = feedback_csv_path

        # Initialize CSV files with headers if they don't exist
        self.initialize_csv(self.rec_csv_path, ["recommendation_id", "user_id", "user_profile", "recommendation"])
        self.initialize_csv(self.feedback_csv_path, ["recommendation_id", "therapist_id", "feedback"])

    def initialize_csv(self, path, headers):
        """
        Create the CSV file with headers if it doesn't exist.
        """
        try:
            with open(path, mode='x', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
        except FileExistsError:
            pass  # File already exists

    def save_to_csv(self, path, row):
        """
        Save a row to the specified CSV file.
        """
        with open(path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)


    def generate_feedback(self, recommendation_text, therapist_id="default_therapist"):
        """
        Generates simulated feedback using AI and saves it to feedback.csv.
        Returns the feedback data dictionary.
        """
        system_prompt = "You are a therapist providing concise feedback on AI-generated recommendations for patients."
        feedback_prompt = f"""Here is the recommendation:

{recommendation_text}

Please provide constructive, practical, and brief feedback that a therapist might record after reviewing this recommendation."""

        # Call AI to generate feedback
        feedback_response = call_groqapi(prompt=feedback_prompt,context_vars=context_vars,system_prompt=system_propmt, model="llama-3.3-70b-versatile")
        cleaned_feedback = feedback_response.strip() if feedback_response else "No feedback provided."

        # Generate dummy recommendation_id for this example if not linked
        # recommendation_id = str(uuid.uuid4())
        recommendation_id = "503fca12-c8b4-4d49-8d38-6bb36b56a3e2"

        # Save feedback to CSV
        self.save_to_csv(self.feedback_csv_path, [recommendation_id, therapist_id, cleaned_feedback])

        # Return structured feedback data
        return {
            "recommendation_id": recommendation_id,
            "therapist_id": therapist_id,
            "feedback": cleaned_feedback
        }


    def recommend(self, user_id,context_vars=None):
        """
        Generate a recommendation for a given user_id and store history with user profile.
        """

        system_prompt = "You are a specialized healthcare AI assistant providing personalized recommendations for patients with sensory processing and behavioral needs."
        
        prompt="""You are a specialized healthcare AI assistant providing personalized, refined recommendations for patients with sensory processing and behavioral needs.

PATIENT PROFILE:
{{patient_profile}}

HISTORY & PREVIOUS RECOMMENDATIONS:
{{patient_history}}

CONTEXTUAL KNOWLEDGE BASE:
{{retrieved_text}}

SENTIMENT ANALYSIS:
{{sentiment_analysis}}

EMOTIONAL STATE DATA:
{{emotional_state}}

BEHAVIORAL ANALYSIS:
{{behavioral_analysis}}

PREVIOUS FEEDBACK & ITERATION (if available):
{{feedback_data}}

if feedback_data is provided Exaplain first the feeddbacks you analyzed for refining recommendation to the patient.

ANALYSIS FRAMEWORK:
Before providing recommendations, conduct a comprehensive analysis using the provided data:

1. **REVIEW HISTORY & PREVIOUS RECOMMENDATIONS:**
   - Analyze patient_history to understand all prior recommendations and interventions provided.
   - Identify what has been tried, the outcomes, and feedback.
   - Ensure recommendations are not repetitive; propose only new, refined, or complementary strategies that build upon what has been done, unless an existing recommendation requires adjustment for improved effectiveness.

2. **INTEGRATE SENTIMENT ANALYSIS DATA:**
   - Use sentiment_analysis to understand current emotional tone and communication patterns.
   - Consider sentiment trends and key phrases indicating the patient's current state.
   - Adapt communication strategies based on sentiment patterns.

3. **UTILIZE EMOTIONAL STATE DATA:**
   - Incorporate emotional_state data to identify primary emotions and triggers.
   - Consider emotional intensity levels when recommending interventions.
   - Address specific emotional regulation needs based on provided emotional data.

4. **APPLY BEHAVIORAL ANALYSIS DATA:**
   - Use behavioral_analysis data to understand behavior patterns, frequency, and triggers.
   - Consider antecedents and consequences when designing interventions.
   - Build on adaptive behaviors and address maladaptive patterns identified.

5. **INTEGRATE FEEDBACK DATA:**
   - Analyze feedback_data to identify therapist or caregiver insights on previous recommendations.
   - Clearly refine and update approaches based on what was successful and what requires modification.
   - When recommendations are improved or adjusted based on feedback, clearly indicate these refinements in the recommendations section to demonstrate continuity and responsiveness to therapist input.

RECOMMENDATION GUIDELINES:
- Analyze the patient's age, behaviors, interests, sensory sensitivities, current therapy approaches, and history.
- Use Contextual Knowledge Base to inform recommendations.
- Ensure recommendations are contextually relevant and avoid repeating identical suggestions unnecessarily.
- Consider developmental appropriateness for the patient's age.
- Leverage the patient's interests to make recommendations more engaging and effective.
- Address specific sensory triggers and sensitivities mentioned.
- Build upon existing therapy approaches when applicable.
- **Prioritize emotional regulation and behavioral support strategies.**
- **Incorporate sentiment analysis to tailor communication and intervention approaches.**
- **Refine recommendations based on previous feedback and outcomes.**
- Provide practical, actionable recommendations that can be implemented by caregivers, teachers, or therapists.

STRUCTURED RECOMMENDATION FORMAT:
if feedback_Data:
    Explain first the feedbacks and how the recommenations you have refined.
CLINICAL ASSESSMENT SUMMARY:
**Primary Concerns Identified:**

[List top 3-5 priority areas based on analysis]

**Strengths & Protective Factors:**

[Identify patient's existing strengths and resources]

**Risk Factors & Triggers:**

[Document key environmental, emotional, or behavioral triggers]

RECOMMENDATIONS:
[List tailored, practical, and contextually appropriate recommendations with clear rationale. For each recommendation improved or adjusted based on therapist feedback, indicate with the prefix "**REFINED BASED ON FEEDBACK:**" followed by the updated recommendation and a brief explanation of the adjustment made. Ensure all recommendations build upon history, maintain continuity of care, and demonstrate responsiveness to feedback.]

EXAMPLE:
**REFINED BASED ON FEEDBACK:** Introduce a weighted lap pad during group activities to improve seated attention. Therapist feedback indicated prior sensory cushion intervention helped briefly but lacked deep pressure input. This refinement incorporates deep pressure to address proprioceptive needs more effectively.

[Continue listing recommendations in this structured manner.]

"""
        if user_id not in self.history:
            self.history[user_id] = []
            self.response_count[user_id] = 0
        # Format prompt with context variables if provided

        if context_vars:
            context_vars["patient_history"] = self.history[user_id]
            prompt = prompt.format(**context_vars)
            user_profile = context_vars.get('patient_profile', 'Unknown')
        else:
            user_profile = 'Unknown'

        recommendation_id = "503fca12-c8b4-4d49-8d38-6bb36b56a3e2"

        if "feedback_data" not in context_vars:
            context_vars["feedback_data"] = []
        matching_feedback = feedback_df[feedback_df['recommendation_id'] == recommendation_id]
        print(matching_feedback)
        for i, row in matching_feedback.iterrows():
            print(context_vars["feedback_data"])
            context_vars["feedback_data"].append(row['feedback'])
        response = call_groqapi(prompt=prompt,context_vars=context_vars,system_prompt=system_prompt, model="llama-3.3-70b-versatile")
        # response = call_openai(prompt,context_vars,system_prompt)
        cleaned_response = response.strip() if response else None
        print("CLeaned Response")
        print(cleaned_response)

        # Store as dictionary with user_profile and recommendation
        if cleaned_response:
            self.history[user_id].append({
                "user_profile": user_profile,
                "recommendation": cleaned_response
            })
            self.save_to_csv(self.rec_csv_path, [recommendation_id, user_id, user_profile, cleaned_response])
            print("History")
            print(self.history[user_id])
            self.response_count[user_id] += 1
            # feedback_data = self.generate_feedback(cleaned_response, therapist_id="therapist123")
            # feedback_data["recommendation_id"] = recommendation_id 

            # self.save_to_csv(self.feedback_csv_path, [recommendation_id, feedback_data["therapist_id"], feedback_data["feedback"]])

        if self.response_count[user_id] == 3:
            summarizer = Summarizer()
            summarize_content = summarizer.analyze(self.history[user_id])
            self.history[user_id]=[]
            self.history[user_id].append({"summarized_content":summarize_content})
            print("History")
            print(self.history[user_id])
            self.response_count[user_id]=0
            
        print(self.response_count[user_id])
        return response
