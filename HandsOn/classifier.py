from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq


# CONFIG

client = Groq(api_key="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")

app = FastAPI(title="LLM Hands-on API")


# REQUEST MODELS

class FeedbackRequest(BaseModel):
    feedback: str

class InfoExtractionRequest(BaseModel):
    text: str


# ZERO-SHOT SENTIMENT

@app.post("/sentiment/zero-shot")
def zero_shot_sentiment(req: FeedbackRequest):
    prompt = f"""
You are an expert customer sentiment classification system.

Classify the customer feedback into exactly ONE of the following categories:
- Positive
- Neutral
- Negative

Rules:
- Respond with only ONE word
- Do not explain your reasoning

Customer Feedback:
"{req.feedback}"

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return {"sentiment": response.choices[0].message.content.strip()}


# FEW-SHOT SENTIMENT

@app.post("/sentiment/few-shot")
def few_shot_sentiment(req: FeedbackRequest):
    prompt = f"""
You are an expert customer sentiment classification system.

Classify the customer feedback into exactly ONE of the following categories:
- Positive
- Neutral
- Negative

Rules:
- Respond with only ONE word
- Do not explain your reasoning

Examples:
Customer Feedback: "The app is amazing and very easy to use."
Answer: Positive

Customer Feedback: "The app crashes frequently and is unusable."
Answer: Negative

Customer Feedback: "The app works as expected, nothing special."
Answer: Neutral

Customer Feedback:
"{req.feedback}"

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return {"sentiment": response.choices[0].message.content.strip()}


# INFORMATION EXTRACTION (HANDS-ON TASK)

@app.post("/extract-info")
def extract_info(req: InfoExtractionRequest):
    prompt = f"""
Extract information from the text below.

Return ONLY valid JSON with exactly these keys:
- name
- company
- role
- start_date
- location

Rules:
- Do not add extra fields
- Do not explain anything
- If a field is missing, return null

Text:
"{req.text}"

JSON:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


@app.get("/")
def health():
    return {"status": "API running"}
