import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# --- Load .env ---
load_dotenv()

# --- Page setup (wide, no title) ---
st.set_page_config(page_title="Qwen Streaming Chat", layout="wide")

st.markdown(
    """
    <style>
        .chat-tip {
            border: 1px dashed rgba(148, 163, 184, 0.45);
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0 1rem 0;
            color: #cbd5e1;
        }
        .stSidebar .stButton button {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Fixed rules (only what you asked) ---
CHUNK_SIZE_TOKENS = 8192
OVERLAP_TOKENS = 200
TIKTOKEN_ENCODING = "cl100k_base"
DEFAULT_MODEL = "qwen3.5-plus"


@st.cache_resource
def get_client():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )


@st.cache_resource
def get_encoder():
    return tiktoken.get_encoding(TIKTOKEN_ENCODING)


def count_tokens(text: str) -> int:
    enc = get_encoder()
    return len(enc.encode(text))


def split_with_overlap_8192_200(text: str) -> list[str]:
    enc = get_encoder()
    toks = enc.encode(text)
    if not toks:
        return [""]

    step = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS
    chunks = []
    for start in range(0, len(toks), step):
        end = min(start + CHUNK_SIZE_TOKENS, len(toks))
        chunks.append(enc.decode(toks[start:end]))
        if end == len(toks):
            break
    return chunks


def create_chat(title: str = "New chat") -> dict:
    return {"title": title, "messages": []}


def get_chat_title(messages: list[dict]) -> str:
    for message in messages:
        if message["role"] == "user" and message["content"].strip():
            content = message["content"].strip().replace("\n", " ")
            return content[:40] + ("..." if len(content) > 40 else "")
    return "New chat"


client = get_client()

# --- Session state ---
if "chats" not in st.session_state:
    first_chat_id = str(uuid.uuid4())
    st.session_state.chats = {first_chat_id: create_chat("New chat")}
    st.session_state.chat_order = [first_chat_id]
    st.session_state.active_chat_id = first_chat_id

current_chat = st.session_state.chats[st.session_state.active_chat_id]

if not client:
    st.warning("DASHSCOPE_API_KEY belum diatur. Set API key di environment agar chat bisa digunakan.")


# --- Sidebar ---
with st.sidebar:
    st.subheader("💬 Chats")

    if st.button("+ New chat", use_container_width=True):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = create_chat("New chat")
        st.session_state.chat_order.insert(0, new_chat_id)
        st.session_state.active_chat_id = new_chat_id
        st.rerun()

    for chat_id in st.session_state.chat_order:
        chat = st.session_state.chats[chat_id]
        if st.button(
            chat["title"],
            key=f"chat_{chat_id}",
            use_container_width=True,
            type="primary" if chat_id == st.session_state.active_chat_id else "secondary",
        ):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    st.divider()
    st.caption("⚙️ Session settings")
    st.caption(f"Model: `{DEFAULT_MODEL}` (locked)")
    enable_thinking = st.toggle("Enable thinking", value=True)

    if st.button("Clear current chat", use_container_width=True):
        st.session_state.chats[st.session_state.active_chat_id]["messages"] = []
        st.session_state.chats[st.session_state.active_chat_id]["title"] = "New chat"
        st.rerun()

current_chat = st.session_state.chats[st.session_state.active_chat_id]


if not current_chat["messages"]:
    st.markdown(
        '<div class="chat-tip">Mulai percakapan dengan menulis pertanyaan di bawah. Untuk teks panjang, app akan otomatis memecahnya menjadi beberapa chunk token.</div>',
        unsafe_allow_html=True,
    )

# --- Render history ---
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input ---
user_text = st.chat_input("Tulis pertanyaan...")

if user_text:
    if not client:
        st.error("Chat tidak dapat diproses karena API key belum tersedia.")
        st.stop()

    # Only split if user_text > 8192 tokens; otherwise keep as one message.
    if count_tokens(user_text) > CHUNK_SIZE_TOKENS:
        chunks = split_with_overlap_8192_200(user_text)
        for chunk in chunks:
            current_chat["messages"].append({"role": "user", "content": chunk})
            with st.chat_message("user"):
                st.markdown(chunk)
    else:
        current_chat["messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

    current_chat["title"] = get_chat_title(current_chat["messages"])

    # --- Assistant streaming ---
    with st.chat_message("assistant"):
        thinking_expander = st.expander("Thinking", expanded=True)
        thinking_box = thinking_expander.empty()
        answer_box = st.empty()

        thinking_text = ""
        answer_text = ""

        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=current_chat["messages"],
            extra_body={"enable_thinking": enable_thinking},
            stream=True,
        )

        for chunk in completion:
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            content = getattr(delta, "content", None)

            if reasoning:
                thinking_text += reasoning
                thinking_box.markdown(thinking_text)

            if content:
                answer_text += content
                answer_box.markdown(answer_text)

        current_chat["messages"].append({"role": "assistant", "content": answer_text})
