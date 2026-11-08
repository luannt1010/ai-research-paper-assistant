import os
import requests
import streamlit as st

class APIClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.ingest_file_url = f"{base_url}/ingestion/ingest_file"
        self.ingest_dir_url = f"{base_url}/ingestion/ingest_dir"
        self.gen_url = f"{base_url}/generation/generate"
        self.session = requests.session()

    def ingest_file(self, file_paths: list[str]):
        response = self.session.post(url=self.ingest_file_url, json={"paths": file_paths})
        response.raise_for_status()
        return response.json()

    def ingest_dir(self, dir_path: str):
        response = self.session.post(url=self.ingest_dir_url, json={"dir_path": dir_path})
        response.raise_for_status()
        return response.json()

    def generate(self, query: str):
        response = self.session.post(url=self.gen_url, json={"question": query})
        response.raise_for_status()
        return response.json()


def get_api_client() -> APIClient:
    """Giữ lại client dùng địa chỉ mặc định qua các lần rerun."""
    if "api_client" not in st.session_state:
        # Đọc API_BASE_URL đã khai báo, nếu k có dùng mặc định local
        st.session_state.api_client = APIClient(os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    return st.session_state.api_client


def call_api(action, *args):
    """Xử lý lỗi chung cho các phương thức của APIClient."""
    try:
        return action(*args)
    except requests.HTTPError as error:
        response = error.response
        try:
            detail = response.json().get("detail", response.reason)
        except ValueError:
            detail = response.reason
        st.error(f"API error ({response.status_code}): {detail}")
    except requests.exceptions.JSONDecodeError:
        st.error("The API returned an invalid JSON response.")
    except requests.RequestException as error:
        st.error(f"Could not reach the API. Check that the backend is running. {error}")
    return None


# Widget trong fragment chỉ chạy lại phần nhập tài liệu, không ngắt luồng chat.
@st.fragment
def render_ingestion(client: APIClient) -> None:
    st.subheader("Add documents")
    st.caption("Enter paths on the backend machine. Only PDF documents are supported.")
    mode = st.radio("Document source", ["PDF files", "Folder"], horizontal=True)

    # Form gom các thay đổi thành một lần gửi, tránh gọi API khi đang nhập.
    with st.form("ingestion_form"):
        if mode == "PDF files":
            path_input = st.text_area(
                "PDF paths, one per line (up to 10)",
                placeholder="D:/papers/paper_1.pdf\nD:/papers/paper_2.pdf",
            )
        else:
            path_input = st.text_input("Folder path", placeholder="D:/papers")
        submitted = st.form_submit_button("Add documents")

    if not submitted:
        return

    if mode == "PDF files":
        paths = [line.strip() for line in path_input.splitlines() if line.strip()]
        if not 1 <= len(paths) <= 10:
            st.warning("Please enter between 1 and 10 PDF paths.")
            return
        action, argument = client.ingest_file, paths
    else:
        if not path_input.strip():
            st.warning("Please enter a folder path.")
            return
        action, argument = client.ingest_dir, path_input.strip()

    with st.spinner("Processing documents..."):
        response = call_api(action, argument)
    if response is None:
        return

    # Hai endpoint có cấu trúc khác nhau; dùng chung phần hiển thị kết quả.
    results = response["results"] if mode == "PDF files" else [response]
    for result in results:
        path = result.get("path", result.get("directory"))
        if result["status"] == "success":
            chunks = result.get("num_chunks", result.get("total_chunks"))
            st.success(f"{path}: ingested {chunks} chunks.")
        else:
            st.error(f"{path}: {result.get('error') or 'Could not ingest this document.'}")


def render_chat(client: APIClient) -> None:
    # Streamlit chạy lại script khi tương tác; lịch sử phải nằm trong session_state.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your documents...", key="chat_question")
    if not question or not question.strip():
        return

    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            response = call_api(client.generate, question)
        if response is None:
            return

        result = response["results"]
        if result["status"] != "success":
            st.error(result.get("error") or "Could not generate an answer.")
            return
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            st.error("The API returned an empty answer.")
            return

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def render_clear_history(placeholder, disabled: bool = False) -> None:
    # Hai key cho phép thay nút đang khóa bằng nút hoạt động trong cùng một lượt chạy.
    if placeholder.button(
        "Clear chat history",
        key="clear_history_busy" if disabled else "clear_history",
        disabled=disabled,
    ):
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    client = get_api_client()
    # Câu hỏi đã có trong session_state trước khi script chạy lại sau khi gửi.
    is_generating = bool(st.session_state.get("chat_question"))

    with st.sidebar:
        render_ingestion(client)
        st.divider()
        clear_history = st.empty()
        render_clear_history(clear_history, disabled=is_generating)

    st.title("📚 Research Paper Chat")
    st.caption(
        "Answers are based on your ingested documents. "
        "Each question is answered independently; previous messages are not used as context."
    )
    try:
        render_chat(client)
    finally:
        if is_generating:
            render_clear_history(clear_history)


if __name__ == "__main__":
    main()
