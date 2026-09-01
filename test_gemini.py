from agents.ai_quality_advisor_agent import AIQualityAdvisorAgent


print("Testing Gemini API...")

try:

    advisor = AIQualityAdvisorAgent()

    test_report = {
        "health_score": 85,
        "total_issues": 1,
        "issues": [
            {
                "type": "Missing Values",
                "column": "Age",
                "count": 10,
                "percentage": 5.0,
                "severity": "Low",
                "recommendation": "Use median imputation.",
                "confidence": 97
            }
        ]
    }

    result = advisor.generate_advice(
        test_report
    )

    print("\n==============================")
    print("GEMINI RESULT")
    print("==============================")

    print(result)

except Exception as e:

    print("\n❌ Gemini test failed:")
    print(e)