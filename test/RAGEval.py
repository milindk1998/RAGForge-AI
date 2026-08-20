from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Retrieval.readEmbedQuery import read_embed_query

from deepeval.models.base_model import DeepEvalBaseLLM
import os
from langchain_groq import ChatGroq
from langsmith import traceable

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "").strip()
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").strip().lower()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "genai-rag-eval").strip()
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT


@traceable(name="AI Eval")

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-safeguard-20b", temperature=0.0)

    def load_model(self):
        return self.model

    def get_model_name(self):
        return "OpenAI GPT-OSS Safeguard 20B"

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

judge = GroqJudge()

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=judge,
    include_reason=True
)
answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True
)
# contextual_relevancy = ContextualRelevancyMetric(
#     threshold=0.7,
#     model=judge,
#     include_reason=True
# )

query = str(input("\nPlease enter question to evaluate?\n"))

actual_output, retrieval_context = read_embed_query(pdf_path=str(Path(__file__).resolve().parents[1] / "document" / "companyPolicy.pdf"))

test_case = LLMTestCase(
    input=query,
    actual_output=actual_output,
    retrieval_context=retrieval_context,
)

metrics = [faithfulness, answer_relevancy]

evaluate(
    test_cases=[test_case],
    metrics=metrics
)