import os
import logging
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from datetime import datetime, timedelta
from fpdf import FPDF
from dotenv import load_dotenv
from llm_service import call_gemini, call_groqapi, call_openai
from markdown import markdown
from bs4 import BeautifulSoup
import html2text
import re
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load environment variables
load_dotenv()
MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME: str = os.getenv("MONGO_DB_NAME", "recommendation_db")
COLLECTION_NAME: str = os.getenv("MONGO_COLLECTION_NAME", "recommendation_logs")
DEFAULT_OUTPUT_DIR: str = "./reports"

def markdown_to_text(markdown_string: str) -> str:
    html = markdown(markdown_string)
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.body_width = 0 
    return text_maker.handle(html).strip()

class ReportGenerator:
    """
    Handles fetching recommendations, summarizing them, and generating PDF reports.
    """

    def __init__(
        self,
        mongo_uri: str = MONGO_URI,
        db_name: str = DB_NAME,
        collection_name: str = COLLECTION_NAME
    ):
        try:
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection: Collection = self.db[collection_name]
            logging.info("Connected to MongoDB at %s", mongo_uri)
        except Exception as e:
            logging.error("Failed to connect to MongoDB: %s", e)
            raise

    def fetch_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Fetch recommendations from MongoDB between start_date and end_date.
        """
        try:
            query = {
                "date": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            records = list(self.collection.find(query))
            logging.info("Fetched %d records from %s to %s", len(records), start_date, end_date)
            return records
        except Exception as e:
            logging.error("Error fetching data: %s", e)
            return []

    def generate_summary(self, user_id: str, recommendations: List[Dict[str, Any]]) -> str:
        """
        Generate a summary report for a single user from recommendations.
        """
        combined_text = "\n\n".join(rec.get("recommendation", "") for rec in recommendations)
        logging.info(f"Recommendation history for user {user_id}:\n{combined_text}")
        system_prompt = (
            "You are a clinical assistant generating a neurological recommendation summary report."
        )
        prompt = (
            f"Patient Report for User ID: {user_id}\n"
            "Below is the compiled history of recommendations for this patient during the selected period:\n\n"
            f"{combined_text}\n"
            "Use the recommendation history and give the user a summarized report of there recommendations.\n"
            "Format the report using Markdown with the following guidelines:\n"
            "- Use '#', '##', or '###' for section headings.\n"
            "- Use bullet points for lists where appropriate.\n"
            "- Use bold for section titles and important terms.\n"
            "- Use paragraphs for narrative content.\n"
            "- Do not repeat any section heading or content.\n"
            "DO not include subheading only include the main headings.\n"
            "Generate a professional report that is concise and informative.\n"
            "Ensure the language is clinical, objective, and suitable for inclusion in a medical record. "
        )
        try:
            summary = call_groqapi(
                prompt=prompt,
                system_prompt=system_prompt,
                model="llama-3.3-70b-versatile"
            )
            logging.info(f"Generated summary for user:{user_id} is : {summary}")
            return summary
        except Exception as e:
            logging.error("Error generating summary for user %s: %s", user_id, e)
            return "Summary generation failed."

    def export_pdf(self, user_id: str, summary: str) -> str:
        """
        Export a Markdown summary into a PDF file in a temporary location, rendering bold and headings.
        Returns the temp file path.
        """
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt=f"Therapy Recommendation Report - User: {user_id}", ln=True, align="C")
            pdf.ln(10)

            lines = summary.splitlines()
            for line in lines:
                if line.startswith("# "):
                    pdf.set_font("Arial", "B", 14)
                    pdf.multi_cell(0, 10, line[2:].strip())
                    pdf.set_font("Arial", size=10)
                elif line.startswith("## "):
                    pdf.set_font("Arial", "B", 12)
                    pdf.multi_cell(0, 8, line[3:].strip())
                    pdf.set_font("Arial", size=10)
                elif line.startswith("### "):
                    pdf.set_font("Arial", "B", 11)
                    pdf.multi_cell(0, 7, line[4:].strip())
                    pdf.set_font("Arial", size=10)
                else:
                    bold_pattern = r"(\*\*|__)(.*?)\1"
                    parts = re.split(bold_pattern, line)
                    i = 0
                    while i < len(parts):
                        if i+2 < len(parts) and (parts[i+1] == "**" or parts[i+1] == "__"):
                            pdf.set_font("Arial", "B", 10)
                            pdf.write(5, parts[i+2])
                            pdf.set_font("Arial", size=10)
                            i += 3
                        else:
                            clean = parts[i].replace("**", "").replace("__", "")
                            pdf.write(5, clean)
                            i += 1
                    pdf.ln(7)
            # Use a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                pdf.output(tmpfile.name)
                return tmpfile.name
        except Exception as e:
            logging.error("Error exporting PDF for user %s: %s", user_id, e)
            return ""

    def generate_reports_for_period(self, start_date_str: str, end_date_str: str, user_id: Optional[str] = None) -> Dict[str, str]:
        """
        Fetch data, summarize and generate PDF reports for the specified user_id (if provided).
        Returns a dict mapping user_id to report file path.
        """
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
        except ValueError as e:
            logging.error("Invalid date format: %s", e)
            return {}

        # Build query
        query = {
            "date": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        if user_id:
            query["user_id"] = user_id

        try:
            records = list(self.collection.find(query))
            logging.info("Fetched %d records for user %s from %s to %s", len(records), user_id, start_date, end_date)
        except Exception as e:
            logging.error("Error fetching data: %s", e)
            return {}

        if not records:
            logging.warning("No data found in the given date range for user %s.", user_id)
            return {}

        # Group recommendations by user_id (should be only one if user_id is provided)
        user_recs: Dict[str, List[Dict[str, Any]]] = {}
        for rec in records:
            uid = rec.get("user_id", "unknown_user")
            user_recs.setdefault(uid, []).append(rec)

        # Generate summary and PDF for each user
        report_paths: Dict[str, str] = {}
        for uid, recs in user_recs.items():
            summary = self.generate_summary(uid, recs)
            path = self.export_pdf(uid, summary)
            if path:
                report_paths[uid] = path

        return report_paths

# if __name__ == "__main__":
#     # Example usage
#     generator = ReportGenerator()
#     reports = generator.generate_reports_for_period("2025-07-01", "2025-07-25")

#     if reports:
#         for user, path in reports.items():
#             logging.info("Report generated for user %s: %s", user, path)
#     else:
#         logging.info("No reports generated.")
