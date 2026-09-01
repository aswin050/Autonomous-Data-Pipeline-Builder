import os
import json
from dotenv import load_dotenv
from google import genai


load_dotenv()


class AIQualityAdvisorAgent:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash"
        )


    def generate_advice(self, quality_report):

        if not quality_report:
            return {
                "status": "error",
                "message": "Quality report is empty."
            }


        # Convert dictionary into readable JSON
        report_text = json.dumps(
            quality_report,
            indent=2,
            default=str
        )


        prompt = f"""
You are an AI Data Quality Advisor
for an autonomous machine-learning
data pipeline.

Your job is to interpret the automated
data-quality analysis provided below.

IMPORTANT RULES:

1. Do not invent statistics.
2. Do not change any values.
3. Use only the information provided.
4. Clearly distinguish detected issues
   from recommendations.
5. Prioritize High severity issues first.
6. Consider machine-learning risks.
7. Give practical recommendations.
8. Do not perform actual data cleaning.
9. Do not claim that an action was performed
   unless the report explicitly says so.

AUTOMATED DATA QUALITY REPORT:

{report_text}


Provide the following sections:

1. Overall Assessment
2. Critical Issues
3. Medium Priority Issues
4. Low Priority Issues
5. Recommended Actions
6. Potential ML Risks
7. Health Score Explanation
8. Final Recommendation

Keep the response clear and suitable
for displaying in a Streamlit dashboard.
"""


        try:

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            return {
                "status": "success",
                "advice": response.text
            }


        except Exception as e:

            return {
                "status": "error",
                "message": str(e)
            }