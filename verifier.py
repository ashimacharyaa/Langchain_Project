from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import format_evidence

class Claim(BaseModel):
    statement: str

    verdict: Literal[
        "SUPPORTED",
        "UNSUPPORTED",
        "UNCLEAR"
    ]

    source_ids: list[str] = Field(default_factory=list)

    explaination : str

    #defining cerification output
class Verification(BaseModel):
    claims : list[Claim]

#structural output that gemini can understand and provide instead of simple output  

verifier_llm = ChatGoogleGenerativeAI(
    model ="gemini-3.8-flash",
    temperature = 0
).with_structured_output(Verification)

verification_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """ You are strict medical evidence auditor.

Your task:
1. Extract every distinct factual medical claim
   from the answer.
2. Compare each claim against the supplied evidence.
3. Mark each claim:

SUPPORTED:
The evidence directly supports the entire claim.

UNSUPPORTED:
The claim contradicts the evidence, or no relevant
support is present.

UNCLEAR:
The evidence is relevant but insufficient to
establish the claim.

Rules:
- Never use your own medical knowledge as evidence.
- Never invent source IDs.
- Verify numbers, doses, percentages and units exactly.
- Do not treat a source mentioning a topic as proof.
- If a claim has multiple parts, split it.
- Be conservative when evidence is ambiguous.
- Return source IDs only for directly supporting claims.
- An absence of evidence is not proof that a claim
  is medically false.


  Evidence:
  {context}"""
    ),
    (
        "human",
        """Question:
    {question}

    Generated answer:
    {answer}

    Audit every factual medical claim.

"""

    )
])

def verify_answer(question, answer, evidence):
    """ Verify generated claims against retrieved sources."""

    chain = verification_prompt | verifier_llm

    result = chain.invoke({
        "question":question,
        "answer":answer,
        "context":format_evidence(evidence)
    })


    valid_ids = {item['id'] for item in evidence}

    for claim in result.claims:
        claim.source_ids = [
            source_id
            for source_id in claim.source_ids
            if source_id in valid_ids
        ]

        if (
            claim.verdict == "SUPPORTED"
            and not claim.source_ids
        ):
            claim.verdict = "UNCLEAR"

    return result

