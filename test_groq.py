from services.llm_service import get_groq_client

def test_web_search():
    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": (
                    "Search the web for the latest Angular news. "
                    "Summarize the findings and include source URLs."
                ),
            }
        ],
        tools=[
            {"type": "browser_search"}
        ],
        tool_choice="auto",
    )

    print("\n--- Web Search Answer ---\n")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    test_web_search()