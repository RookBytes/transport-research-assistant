from __future__ import annotations

import httpx
import streamlit as st


API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

st.set_page_config(page_title="Transportation Research Assistant", page_icon="🚌")
st.title("Transportation Research Assistant")
st.caption("Source-grounded answers over the Porto route-simulation research project.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.markdown(
                        f"**[{src['id']}] {src['source_path']}:{src['start_line']}-{src['end_line']}** "
                        f"(score {src['score']:.3f})"
                    )
                    st.code(src["text"], language=None)

question = st.chat_input("Ask about the simulation, preprocessing, configuration, or results")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Retrieving evidence..."):
            response = httpx.post(
                f"{API_URL.rstrip('/')}/ask",
                json={"question": question},
                timeout=180.0,
            )
            response.raise_for_status()
            data = response.json()
        answer = data["answer"]
        sources = data.get("sources", [])
    except Exception as exc:
        answer = f"API error: {exc}"
        sources = []

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(
                        f"**[{src['id']}] {src['source_path']}:{src['start_line']}-{src['end_line']}** "
                        f"(score {src['score']:.3f})"
                    )
                    st.code(src["text"], language=None)
