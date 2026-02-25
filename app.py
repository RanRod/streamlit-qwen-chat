import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# --- Load .env ---
load_dotenv()

# --- Page setup (wide, no title) ---
st.set_page_config(page_title="Qwen Streaming Chat", layout="wide")

# --- Fixed rules (only what you asked) ---
CHUNK_SIZE_TOKENS = 8192
OVERLAP_TOKENS = 200
TIKTOKEN_ENCODING = "cl100k_base"
DEFAULT_MODEL = "qwen3.5-plus"

@st.cache_resource
def get_client():
    api_key = os.getenv("DASHSCOPE_API_KEY")
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

client = get_client()

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL

# --- Sidebar ---
with st.sidebar:
    st.subheader("Settings")
    st.session_state.model = st.text_input("Model", st.session_state.model)
    enable_thinking = st.toggle("Enable thinking", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Render history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Input ---
user_text = st.chat_input("Tulis pertanyaan...")

if user_text:
    # Only split if user_text > 8192 tokens; otherwise keep as one message.
    if count_tokens(user_text) > CHUNK_SIZE_TOKENS:
        chunks = split_with_overlap_8192_200(user_text)
        for c in chunks:
            st.session_state.messages.append({"role": "user", "content": c})
            with st.chat_message("user"):
                st.markdown(c)
    else:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

    # --- Assistant streaming ---
    with st.chat_message("assistant"):
        thinking_expander = st.expander("Thinking", expanded=True)
        thinking_box = thinking_expander.empty()
        answer_box = st.empty()

        thinking_text = ""
        answer_text = ""

        completion = client.chat.completions.create(
            model=st.session_state.model,
            messages=st.session_state.messages,
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

        st.session_state.messages.append({"role": "assistant", "content": answer_text})