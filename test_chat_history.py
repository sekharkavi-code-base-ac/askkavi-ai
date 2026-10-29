import uuid

from services.chat_history import (
    create_chat_session,
    save_chat_message,
    load_chat_sessions,
    load_chat_messages,
)

# Create a unique conversation ID.
chat_id = str(uuid.uuid4())

# Create a conversation.
create_chat_session(
    chat_id,
    "Chat History Test",
)

# Save a question.
save_chat_message(
    chat_id,
    "user",
    "What is Python?",
)

# Save an answer.
save_chat_message(
    chat_id,
    "assistant",
    "Python is a programming language.",
    "general_ai",
)

# Retrieve the conversations.
print("\nSaved conversations:")

for chat in load_chat_sessions():
    print(chat["title"], chat["id"])

# Retrieve messages from our test conversation.
print("\nMessages in the test conversation:")

for message in load_chat_messages(chat_id):
    print(
        message["role"],
        message["content"],
        message["source"],
    )