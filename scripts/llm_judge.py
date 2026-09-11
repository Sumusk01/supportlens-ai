import os
import json
from google import genai


MODEL = "gemini-3.6-flash"


def judge_relevance(
    query_message: str,
    historical_customer_message: str,
    historical_support_reply: str,
):
    """
    Judge how useful a historical support case is for answering
    the current customer query.

    Labels:
        0 = Unrelated
        1 = Useful but not an exact/core match
        2 = Direct/core problem match
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set in the environment."
        )

    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1"},
    )

    prompt = f"""
You are evaluating a retrieval system for a customer-support AI agent.

Your task is to judge whether ONE historical customer-support case
is useful evidence for answering a NEW customer query.

Use exactly one of these labels:

2 = DIRECT MATCH
The historical customer has essentially the same core problem as the
new customer. The historical support response is strong evidence for
how the new case could be handled.

1 = USEFUL / SAME DOMAIN
The historical case is not an exact core-problem match, but it is
still useful because it concerns the same problem domain or the
historical support response contains guidance that could reasonably
help with the new query.

0 = UNRELATED
The historical case does not provide useful evidence for the new query.
Shared generic words, device names, or superficial similarity alone
are not enough.

NEW CUSTOMER QUERY:
{query_message}

HISTORICAL CUSTOMER MESSAGE:
{historical_customer_message}

HISTORICAL SUPPORT REPLY:
{historical_support_reply}

Return ONLY JSON with:

{{
  "relevance": 0,
  "rationale": "brief explanation"
}}

The relevance value MUST be an integer: 0, 1, or 2.
The rationale must be no more than two sentences.
"""

    interaction = client.interactions.create(
        model=MODEL,
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {
                    "relevance": {
                        "type": "integer",
                        "enum": [0, 1, 2],
                    },
                    "rationale": {
                        "type": "string",
                    },
                },
                "required": ["relevance", "rationale"],
            },
        },
    )

    result = json.loads(interaction.output_text)

    if result["relevance"] not in (0, 1, 2):
        raise ValueError(
            f"Invalid relevance score: {result['relevance']}"
        )

    return result


if __name__ == "__main__":
    result = judge_relevance(
        query_message="My iPhone battery is draining very quickly",

        historical_customer_message="Battery draining very quickly",

        historical_support_reply=(
            "Thanks for reaching out. "
            "Please send us a DM so we can look into this with you."
        ),
    )

    print("Gemini judge result:")
    print(json.dumps(result, indent=2))