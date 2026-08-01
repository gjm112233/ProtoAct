import time
import streamlit as st
from rag import RagService
import config_data as config

st.title("Biological Experiment Protocol Parsing Assistant")  # Title
st.divider()  # Divider


# st.session_state is a session-state recorder. It is a dictionary that can store chat history.
# Variables stored inside it are not reset when the page refreshes.
if "message" not in st.session_state:
    # Create an instance on the first run
    st.session_state["message"] = [{"role": "assistant", "content": "Hello, how can I help you?"}]


if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()


def preserve_line_breaks(text: str) -> str:
    """
    将普通换行转换为 Markdown 硬换行，
    同时保留 Monitor Condition 之间的空行。
    """
    if not isinstance(text, str):
        return text

    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # 先按照空行划分不同的 Monitor Condition 块
    blocks = text.split("\n\n")

    # 每个块内部的单换行改成 Markdown 硬换行
    formatted_blocks = [
        block.replace("\n", "  \n")
        for block in blocks
    ]

    # 不同块之间仍保留一个空行
    return "\n\n".join(formatted_blocks)


# Each page refresh reruns the current file,
# but st.session_state is not reset.
for message in st.session_state["message"]:

    with st.chat_message(name=message["role"]):

        # Assistant 的结构化结果需要保留换行
        if (
            message["role"] == "assistant"
            and isinstance(message["content"], str)
        ):
            st.markdown(
                preserve_line_breaks(message["content"])
            )

        # 用户消息仍然正常显示
        else:
            st.write(message["content"])


# Provide a user input field at the bottom of the page
prompt = st.chat_input()

if prompt:
    # Output the user's question on the page
    with st.chat_message(name="user"):
        st.write(prompt)

    st.session_state["message"].append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.spinner("AI is thinking..."):
        chain = st.session_state["rag"].chain

        # Streaming output
        answer_stream = chain.stream(
            {"input": prompt},
            config.session_config
        )

        with st.chat_message(name="assistant"):
            # 创建一个可被后续内容覆盖的占位区域
            response_placeholder = st.empty()

            # 先正常进行流式输出
            with response_placeholder:
                res = st.write_stream(answer_stream)

            # 流式输出完成后，重新按照规定格式渲染
            if isinstance(res, str):
                response_placeholder.markdown(
                    preserve_line_breaks(res)
                )
            else:
                response_placeholder.write(res)

        # Save chat history
        st.session_state["message"].append(
            {
                "role": "assistant",
                "content": res
            }
        )