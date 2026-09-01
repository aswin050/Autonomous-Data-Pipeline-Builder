import pandas as pd

from agents.dataset_assistant_agent import (
    DatasetAssistantAgent
)


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    "datasets/train.csv"
)


# =====================================================
# CREATE ASSISTANT
# =====================================================

assistant = DatasetAssistantAgent()


# =====================================================
# ASK QUESTION
# =====================================================

question = (
    "Show me the distribution of Fare."
)
result = assistant.ask(
    df,
    question
)


# =====================================================
# DISPLAY RESULT
# =====================================================

print("\n")
print("=" * 70)
print("🤖 DATASET ASSISTANT")
print("=" * 70)


if result.get("status") == "success":

    print(
        "\nQuestion:"
    )

    print(
        result["question"]
    )

    print(
        "\nAnswer:"
    )

    print(
        result["answer"]
    )

else:

    print(
        "\n❌ Assistant failed:"
    )

    print(
        result.get(
            "message",
            "Unknown error"
        )
    )