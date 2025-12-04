import streamlit as st

st.set_page_config(page_title="RAG Agent UI", page_icon="🤖")

st.title("RAG Agent — Streamlit UI")

st.markdown(
    "Enter a question below and the agent will respond using the indexed context from the vector store."
)

query = st.text_area("Your question", height=160)

try:
    from rag_app import agent  # builds agent on import (expects env vars configured)
except Exception as e:
    st.error(
        "Failed to import agent from `rag_app`. Make sure environment variables are set (GOOGLE_API_KEY, SUPABASE_CONNECTION_STRING) and dependencies are installed.\nError: {}".format(
            e
        )
    )
    st.stop()


def run_agent_stream(query_text: str):
    """Run the agent in streaming mode and yield partial outputs."""
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query_text}]}, stream_mode="values"
    ):
        last_message = event["messages"][-1]
        # Some message objects expose .text, fallback to mapping access
        text = getattr(last_message, "text", None) or last_message.get("text", "")
        yield text


if st.button("Ask"):
    if not query or not query.strip():
        st.warning("Please enter a question before asking the agent.")
    else:
        output = st.empty()
        progress = st.progress(0)
        accumulated = ""
        i = 0
        try:
            for text in run_agent_stream(query):
                accumulated = text
                output.markdown(accumulated)
                i += 1
                # update a simple progress indicator (not exact)
                progress.progress(min(100, i * 5))
        except Exception as e:
            st.error(f"Agent error: {e}")
        else:
            progress.progress(100)
            st.success("Agent finished responding.")
