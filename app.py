import streamlit as st
import requests
import io
import base64
from PIL import Image

st.set_page_config(page_title="DressDreamer", layout="wide")

st.markdown("""
<style>
    /* Global styles */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f2f5;
        color: #333;
    }
    
    /* Header */
    .stApp > header {
        background: linear-gradient(135deg, #ff5f6d, #ffc371);
        color: white;
        border-bottom: none;
        padding: 20px;
        text-align: center;
    }
    
    /* Main content */
    .stApp > div:nth-child(2) {
        padding-top: 20px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 2px solid #e0e0e0;
        padding: 20px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: 500;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #ff4d4d;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #ff6f61;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        background-color: #ffffff;
    }
    
    /* Chat messages */
    .chat-message {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .chat-message .message {
        color: #333;
    }
    
    /* Images */
    img {
        border-radius: 10px;
    }

    /* Vibrant title */
    .vibrant-title {
        font-family: 'Arial', sans-serif;
        font-size: 36px;
        font-weight: 700;
        color: #ff5f6d;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Chat Input */
    .stChatInput>div>input {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 20px;
        padding: 10px 24px;
    }
    
    /* Gradient text for chat messages */
    .gradient-text {
        background: linear-gradient(45deg, #405de6, #5851db, #833ab4, #c13584, #e1306c, #fd1d1d);
        color: transparent;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    
</style>
""", unsafe_allow_html=True)

# Vibrant title
st.markdown("<h1 class='vibrant-title'>DressDreamer</h1>", unsafe_allow_html=True)

IMAGE_API_ENDPOINT = "https://f903-34-125-138-247.ngrok-free.app/fashion"

def generate_image(file):
    files = {'file': (file.name, file.getvalue(), file.type)}
    response = requests.post(IMAGE_API_ENDPOINT, files=files)
    if response.status_code == 200:
        image_data = base64.b64decode(response.json()["image"])
        return Image.open(io.BytesIO(image_data))
    else:
        st.error(f"Error: {response.status_code}")
        return None

def send_text_get_response(prompt):
    # Placeholder function for generating a response
    return "패션 관련 정보를 찾고 있습니다..."

def main():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if "content" in message:
                st.markdown(f"<span class='gradient-text'>{message['content']}</span>", unsafe_allow_html=True)
            if "image" in message:
                st.image(message['image'])
    st.sidebar.title("DressDreamer Info")
    st.sidebar.info(
        "DressDreamer는 외모 분석을 통한 개인화된 패션 추천 시스템입니다."
    )

    st.sidebar.title("사용 방법")
    st.sidebar.markdown(
        """
        **패션 추천**: 이미지를 업로드하고 '**패션 추천해줘**'를 입력하면 패션 추천 이미지를 생성해줍니다.
        """
    )
    st.sidebar.title("개발자 정보")
    st.sidebar.markdown(
        """
        - **개발자**: 정강빈
        - **버전**: 2.0.0
        """
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("이미지를 업로드 하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.session_state.uploaded_image = image
    st.markdown("</div>", unsafe_allow_html=True)
    
    if prompt := st.chat_input('질문을 입력하세요'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(f"<span class='gradient-text'>{prompt}</span>", unsafe_allow_html=True)

        with st.chat_message('assistant'):
            if "패션 추천해줘" in prompt:
                if uploaded_file is not None:
                    with st.spinner("이미지 생성 중..."):
                        image = generate_image(uploaded_file)
                        if image:
                            st.session_state.messages.append({"role": "assistant", "image": image})
                            st.image(image)
                            response = "이미지가 생성되었습니다."
                        else:
                            response = "이미지 생성에 실패했습니다."
                else:
                    response = "이미지를 업로드 해주세요."
            else:
                with st.spinner('답변 생성 중...'):
                    response = send_text_get_response(prompt)
            
            st.markdown(f"<span class='gradient-text'>{response}</span>", unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()
