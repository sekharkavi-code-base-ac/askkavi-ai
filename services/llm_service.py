import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    return Groq(api_key=api_key)


def generate_answer(question, context):
    """Generate an answer using the provided document context."""

    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AskKavi AI, a document assistant. "
                    "Answer the user's question using only the "
                    "provided document context. "
                    "If the answer is not available in the context, "
                    "say that the document does not contain enough "
                    "information. Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Document context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def check_document_answerability(question, context):
    """
    Check whether the document contains enough information
    to answer the user's question.
    """

    if not context or not context.strip():
        return False

    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You determine whether a document contains "
                    "enough information to answer a question.\n\n"
                    "Reply YES if the context provides sufficient "
                    "information to answer the question.\n"
                    "Reply NO if the answer is missing, the context "
                    "is only loosely related, or there is not enough "
                    "information to answer reliably.\n\n"
                    "Consider only the supplied context. Do not use "
                    "your general knowledge. Reply with exactly "
                    "YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Document context:\n{context}"
                ),
            },
        ],
        temperature=0,
    )

    result = response.choices[0].message.content.strip().upper()

    return result == "YES"


def generate_general_answer(question):
    """Answer a question using the LLM's general knowledge."""

    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AskKavi AI, a helpful AI assistant. "
                    "Answer the user's question clearly and accurately "
                    "using your general knowledge. "
                    "If you are uncertain, explain the uncertainty. "
                    "Do not claim that you searched the internet."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def answer_question_with_fallback(question, context):
    """
    First try to answer from the document.
    If the document does not contain enough information,
    fall back to the LLM's general knowledge.

    Returns:
        answer: Generated answer.
        source: Either 'document' or 'general_ai'.
    """

    if check_document_answerability(question, context):
        answer = generate_answer(question, context)
        return answer, "document"

    answer = generate_general_answer(question)
    return answer, "general_ai"


def generate_answer_stream(question, context):
    """Stream an answer based only on the document context."""

    client = get_groq_client()

    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AskKavi AI, a document assistant. "
                    "Answer using only the provided document context. "
                    "If the answer is unavailable, say that the document "
                    "does not contain enough information. "
                    "Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Document context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content

        if content:
            yield content


def generate_general_answer_stream(question):
    """Stream an answer using the LLM's general knowledge."""

    client = get_groq_client()

    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AskKavi AI, a helpful AI assistant. "
                    "Answer clearly using your general knowledge. "
                    "If uncertain, explain the uncertainty. "
                    "Do not claim that you searched the internet."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content

        if content:
            yield content


def generate_web_search_answer_stream(question):
    """Stream an answer using Groq's built-in web search."""

    client = get_groq_client()

    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AskKavi AI, a helpful AI assistant with web search. "
                    "Use the browser_search tool when the question requires "
                    "current information or reliable external sources. "
                    "Summarize the information clearly and accurately. "
                    "When you use web sources, include their titles and URLs. "
                    "Never claim you searched the web unless you actually did."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        tools=[
            {"type": "browser_search"}
        ],
        tool_choice="auto",
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if content:
            yield content

def generate_web_search_answer(question):
    """Answer questions using Groq's built-in browser search."""

    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AskKavi AI, an AI assistant with web search. "
                    "Search the web for current or time-sensitive questions. "
                    "Use reliable sources and verify dates carefully. "
                    "Never present outdated information as current. "
                    "Include source URLs in your answer when available. "
                    "If information cannot be verified, clearly say so."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        tools=[
            {"type": "browser_search"}
        ],
        tool_choice="required",
        temperature=0.2,
        max_completion_tokens=2048,
        stream=False,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("Web search returned an empty answer.")

    return answer