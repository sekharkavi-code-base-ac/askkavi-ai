import uuid
import streamlit as st

from services.document_processor import (
    extract_text_from_pdf,
    split_text_into_chunks,
)

from services.embeddings import generate_embeddings

from services.vector_store import (
    store_document_chunks,
    search_document_chunks,
)

from services.llm_service import (
    generate_answer_stream,
    generate_general_answer_stream,
    generate_web_search_answer_stream,
    check_document_answerability,
    generate_web_search_answer
)

from services.chat_history import (
    create_chat_session,
    save_chat_message,
    update_chat_session,
    load_chat_sessions,
    load_chat_messages,
    delete_chat_session
)
from streamlit_mic_recorder import speech_to_text

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AskKavi AI",
    page_icon="💬",
    layout="wide",
)


# Keep the microphone docked beside Streamlit's bottom chat input.
# The mic component is visually positioned over the input row so its location
# does not change when Streamlit reruns after a voice recording or AI response.
st.markdown(
    """
    <style>
    div[data-testid="stElementContainer"]:has(iframe[src*="streamlit_mic_recorder"]),
    div.element-container:has(iframe[src*="streamlit_mic_recorder"]) {
        position: fixed !important;
        bottom: 48px !important;
        right: clamp(145px, 11vw, 175px) !important;
        width: 50px !important;
        min-width: 50px !important;
        max-width: 50px !important;
        height: 50px !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 1005 !important;
        overflow: visible !important;
        background: transparent !important;
    }

    div[data-testid="stElementContainer"]:has(iframe[src*="streamlit_mic_recorder"]) iframe,
    div.element-container:has(iframe[src*="streamlit_mic_recorder"]) iframe {
        display: block !important;
        width: 50px !important;
        min-width: 50px !important;
        max-width: 50px !important;
        height: 50px !important;
        border: 0 !important;
    }

    /* Reserve space for the mic so typed text does not run underneath it. */
    [data-testid="stChatInput"] textarea,
    textarea[aria-label="Message AskKavi AI..."] {
        padding-right: 105px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 2. Load conversations from Supabase
# --------------------------------------------------

if "chats" not in st.session_state:

    st.session_state.chats = {}

    try:
        saved_chats = load_chat_sessions()

        for chat in saved_chats:

            chat_id = str(chat["id"])

            st.session_state.chats[chat_id] = {
                "title": chat["title"],
                "messages": [],
                "document_id": chat.get("document_id"),
                "document_name": chat.get("document_name"),
            }

    except Exception as error:
        st.error(f"Could not load chat history: {error}")


if "current_chat_id" not in st.session_state:

    if st.session_state.chats:
        st.session_state.current_chat_id = next(
            iter(st.session_state.chats)
        )
    else:
        st.session_state.current_chat_id = None


# --------------------------------------------------
# 3. Create a new conversation
# --------------------------------------------------

def create_new_chat():

    chat_id = str(uuid.uuid4())

    new_chat = {
        "title": "New chat",
        "messages": [],
        "document_id": None,
        "document_name": None,
    }

    try:
        create_chat_session(
            chat_id,
            "New chat",
        )

        st.session_state.chats[chat_id] = new_chat

        st.session_state.current_chat_id = chat_id

    except Exception as error:
        st.error(f"Could not create a new chat: {error}")


# Create the initial conversation if none exists.
if not st.session_state.chats:

    create_new_chat()


# --------------------------------------------------
# 4. Load messages when switching conversations
# --------------------------------------------------

def ensure_chat_messages_loaded(chat_id):

    chat = st.session_state.chats[chat_id]

    if chat.get("messages_loaded", False):
        return

    messages = load_chat_messages(chat_id)

    chat["messages"] = [
        {
            "role": message["role"],
            "content": message["content"],
            "source": message.get("source"),
        }
        for message in messages
    ]

    chat["messages_loaded"] = True


current_chat_id = st.session_state.current_chat_id

if current_chat_id:
    try:
        ensure_chat_messages_loaded(current_chat_id)
    except Exception as error:
        st.error(f"Could not load conversation messages: {error}")


# --------------------------------------------------
# 5. Sidebar: chat history
# --------------------------------------------------

with st.sidebar:

    st.title("💬 AskKavi AI")

    if st.button(
        "＋ New chat",
        use_container_width=True,
    ):
        create_new_chat()
        st.rerun()

    st.divider()

    st.subheader("Chat history")

    # Display conversations from newest to oldest.
    chat_items = list(st.session_state.chats.items())

    for chat_id, chat in reversed(chat_items):

        is_current_chat = (
            chat_id == st.session_state.current_chat_id
        )

        button_label = (
            f"● {chat['title']}"
            if is_current_chat
            else chat["title"]
        )

        # Conversation selection and delete button.
        chat_column, delete_column = st.columns([0.82, 0.18])

        with chat_column:

            if st.button(
                button_label,
                key=f"chat_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()

        with delete_column:

            if st.button(
                "🗑️",
                key=f"delete_{chat_id}",
                help="Delete this conversation",
                use_container_width=True,
            ):
                st.session_state["chat_to_delete"] = chat_id

        # Confirm permanent deletion.
        if st.session_state.get("chat_to_delete") == chat_id:

            st.warning(
                f"Delete '{chat['title']}' permanently?"
            )

            confirm_column, cancel_column = st.columns(2)

            with confirm_column:

                if st.button(
                    "Delete",
                    key=f"confirm_delete_{chat_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        delete_chat_session(chat_id)

                        del st.session_state.chats[chat_id]

                        st.session_state.pop("chat_to_delete", None)

                        if (
                            st.session_state.current_chat_id == chat_id
                            or not st.session_state.chats
                        ):
                            create_new_chat()

                        st.rerun()

                    except Exception as error:
                        st.error(
                            f"Could not delete the conversation: {error}"
                        )

            with cancel_column:

                if st.button(
                    "Cancel",
                    key=f"cancel_delete_{chat_id}",
                    use_container_width=True,
                ):
                    st.session_state.pop("chat_to_delete", None)
                    st.rerun()

    st.divider()

    st.caption("Ask questions about your documents.")






# --------------------------------------------------
# 6. Main chat window
# --------------------------------------------------

current_chat_id = st.session_state.current_chat_id

current_chat = st.session_state.chats[current_chat_id]

st.title("AskKavi AI")

if current_chat.get("document_name"):

    st.caption(
        f"📄 Current document: "
        f"{current_chat['document_name']}"
    )

else:

    st.caption(
        "Ask a question or attach a PDF to get started."
    )


# Display existing messages.
for message in current_chat["messages"]:

    with st.chat_message(message["role"]):

        if (
            message["role"] == "assistant"
            and message.get("source") == "document"
        ):
            st.caption("📄 Answered using your document")

        elif (
            message["role"] == "assistant"
            and message.get("source") == "general_ai"
        ):
            st.caption("🤖 Answered using general AI knowledge")

        st.markdown(message["content"])


# --------------------------------------------------
# 7. Chat input with PDF attachment and voice input
# --------------------------------------------------

# The CSS above docks this mic component beside the native bottom chat input.
# Keep st.chat_input at the main page level so Streamlit pins it to the bottom.
voice_text = speech_to_text(
    start_prompt="🎤",
    stop_prompt="⏹️",
    language="en",
    just_once=True,
    use_container_width=False,
    key=f"voice_input_{current_chat_id}",
)

# Keep this at the main page level (not inside columns or a container).
prompt = st.chat_input(
    "Message AskKavi AI...",
    accept_file=True,
    file_type=["pdf"],
    key=f"chat_input_{current_chat_id}",
)

# Decide which input was submitted.
question = ""
attached_files = []

if prompt is not None:
    # Streamlit chat input may return text or an object containing
    # both text and uploaded files.
    if isinstance(prompt, str):
        question = prompt.strip()
    else:
        question = (prompt.text or "").strip()
        attached_files = prompt.files or []

elif voice_text and voice_text.strip():
    question = voice_text.strip()

# Process only when the user submitted a message or attached a PDF.
if prompt is not None or question:

    # Process uploaded PDFs first.
    for uploaded_file in attached_files:
        try:
            with st.spinner(f"📖 Reading {uploaded_file.name}..."):
                pages = extract_text_from_pdf(uploaded_file)

            if not pages:
                st.warning(
                    f"No readable text was found in {uploaded_file.name}. "
                    "Scanned PDFs may require OCR."
                )
                continue

            full_text = "\n\n".join(page["text"] for page in pages)

            with st.spinner("✂️ Splitting document into chunks..."):
                chunks = split_text_into_chunks(full_text)

            if not chunks:
                st.warning("No text chunks could be created.")
                continue

            with st.spinner("🧠 Generating embeddings..."):
                embeddings = generate_embeddings(chunks)

            document_id = str(uuid.uuid4())

            with st.spinner("☁️ Saving document to Supabase..."):
                saved_count = store_document_chunks(
                    chunks,
                    embeddings,
                    uploaded_file.name,
                    document_id,
                )

            current_chat["document_id"] = document_id
            current_chat["document_name"] = uploaded_file.name

            update_chat_session(
                current_chat_id,
                document_id=document_id,
                document_name=uploaded_file.name,
            )

            st.success(
                f"Processed {uploaded_file.name} "
                f"and saved {saved_count} chunks."
            )

            if current_chat["title"] == "New chat":
                current_chat["title"] = uploaded_file.name
                update_chat_session(
                    current_chat_id,
                    title=uploaded_file.name,
                )

        except Exception as exc:
            st.error(f"Failed to process {uploaded_file.name}.")
            st.exception(exc)

    # Do not call the AI if the user only uploaded a PDF.
    if question:
        # Display the user's question.
        with st.chat_message("user"):
            st.markdown(question)

        # Save the user's question to Supabase.
        try:
            save_chat_message(
                current_chat_id,
                "user",
                question,
            )
        except Exception as error:
            st.error(f"Could not save your question: {error}")

        # Update the in-memory chat history.
        current_chat["messages"].append(
            {
                "role": "user",
                "content": question,
            }
        )

        # Use the first question as the conversation title.
        if current_chat["title"] == "New chat":
            title = question[:50]
            if len(question) > 50:
                title += "..."

            current_chat["title"] = title
            try:
                update_chat_session(
                    current_chat_id,
                    title=title,
                )
            except Exception as error:
                st.warning(
                    f"Your question was saved, but the title "
                    f"could not be updated: {error}"
                )

        # Generate the answer using the current PDF when available;
        # otherwise, use web search.
        with st.chat_message("assistant"):
            try:
                document_id = current_chat.get("document_id")

                if document_id:
                    with st.spinner("Searching your document..."):
                        question_embedding = generate_embeddings(
                            [question]
                        )[0]

                        results = search_document_chunks(
                            question_embedding,
                            match_count=5,
                            document_id=document_id,
                        )

                    context = "\n\n".join(
                        result["content"] for result in results
                    )

                    with st.spinner("Checking document relevance..."):
                        document_has_answer = check_document_answerability(
                            question,
                            context,
                        )

                    if document_has_answer:
                        source = "document"
                        st.caption("📄 Answered using your document")
                        answer = st.write_stream(
                            generate_answer_stream(question, context)
                        )
                    else:
                        source = "web_search"
                        with st.spinner(
                            "🌐 Searching the web and preparing your answer..."
                        ):
                            answer = generate_web_search_answer(question)

                        st.caption("🌐 Answered using web search")
                        st.markdown(answer)

                else:
                    source = "web_search"
                    with st.spinner(
                        "🌐 Searching the web and preparing your answer..."
                    ):
                        answer = generate_web_search_answer(question)

                    st.caption("🌐 Answered using web search")
                    st.markdown(answer)

                # Save the completed answer to Supabase.
                save_chat_message(
                    current_chat_id,
                    "assistant",
                    answer,
                    source,
                )

                # Update the in-memory chat history.
                current_chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "source": source,
                    }
                )

            except Exception as error:
                error_message = str(error)

                if "rate_limit_exceeded" in error_message:
                    st.warning(
                        "The AI service has reached its usage limit. "
                        "Please wait and try again later."
                    )
                else:
                    st.error(
                        f"Sorry, something went wrong: {error_message}"
                    )
