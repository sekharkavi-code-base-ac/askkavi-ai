from services.vector_store import get_supabase_client


def create_chat_session(chat_id, title="New chat"):
    """Create a new conversation in Supabase."""

    supabase = get_supabase_client()

    response = supabase.table("chat_sessions").insert({
        "id": chat_id,
        "title": title,
    }).execute()

    return response.data[0]


def save_chat_message(chat_id, role, content, source=None):
    """Save a user question or assistant answer."""

    supabase = get_supabase_client()

    response = supabase.table("chat_messages").insert({
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "source": source,
    }).execute()

    # Update the conversation's last activity time.
    supabase.table("chat_sessions").update({
        "updated_at": "now()",
    }).eq("id", chat_id).execute()

    return response.data[0]


def update_chat_session(chat_id, title=None,
                        document_id=None, document_name=None):
    """Update conversation details."""

    supabase = get_supabase_client()

    updates = {}

    if title is not None:
        updates["title"] = title

    if document_id is not None:
        updates["document_id"] = document_id

    if document_name is not None:
        updates["document_name"] = document_name

    if not updates:
        return

    supabase.table("chat_sessions").update(
        updates
    ).eq("id", chat_id).execute()


def load_chat_sessions():
    """Retrieve all saved conversations."""

    supabase = get_supabase_client()

    response = (
        supabase.table("chat_sessions")
        .select("*")
        .order("updated_at", desc=True)
        .execute()
    )

    return response.data


def load_chat_messages(chat_id):
    """Retrieve messages belonging to one conversation."""

    supabase = get_supabase_client()

    response = (
        supabase.table("chat_messages")
        .select("*")
        .eq("chat_id", chat_id)
        .order("created_at")
        .order("id")
        .execute()
    )

    return response.data

def delete_chat_session(chat_id):
    """Delete a conversation and its messages from Supabase."""

    supabase = get_supabase_client()

    response = (
        supabase.table("chat_sessions")
        .delete()
        .eq("id", chat_id)
        .execute()
    )

    return response.data

