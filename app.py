import streamlit as st
import torch
import numpy as np
import google.generativeai as genai
from PIL import Image, ImageOps
import mediapipe as mp
import cv2
from tensorflow.keras.models import load_model
import os
import suno
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from model import Generator
import requests
from streamlit_lottie import st_lottie
import io
# Configure page
st.set_page_config(page_title="OmniVerse",page_icon='🤗')
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
st.markdown("""
<style>
.stSpinner > div > div {
    border-top-color: #FF4B4B !important;
}
</style>
""", unsafe_allow_html=True)

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation
lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
lottie_json = load_lottie_url(lottie_url)

# Custom CSS
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-image: linear-gradient(135deg, rgba(221,77,21,1) 11%, rgba(218,148,233,1) 86%);
    }
    .sidebar-title {
        font-size: 30px !important;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .sidebar-subtitle {
        font-size: 20px !important;
        font-weight: bold;
        text-align: center;
        color: #ffd700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .sidebar-text {
        text-align: justify;
        color: #ffffff;
        background-color: rgba(0,0,0,0.1);
        padding: 10px;
        border-radius: 5px;
    }
    .feature-title {
        font-size: 18px !important;
        font-weight: bold;
        color: #ffd700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.markdown('<p class="sidebar-title">🌟 OmniVerse Info</p>', unsafe_allow_html=True)

# Lottie animation
st_lottie(lottie_json, height=200, key="lottie")
st.sidebar.markdown('<p class="sidebar-text">OmniVerse는 Gemini 모델과 FLUX를 활용하여 패션 추천 이미지 생성, 외모 점수 예측, 음악 생성, 이미지 웹툰화, 그리고 이미지 분석 기능을 통합한 시스템입니다. 이 모든 기능은 Gemini 기술을 기반으로 하여, 보다 정교하고 개인화된 경험을 제공합니다.</p>', unsafe_allow_html=True)

st.sidebar.markdown('<p class="sidebar-subtitle">🚀 사용 방법</p>', unsafe_allow_html=True)
features = [
    ("💬 Gemini 챗봇", "다양한 질문에 답변하고 유용한 정보를 제공합니다. 특정 주제에 대한 질문도 가능하니 편하게 이야기해보세요!"),
    ("🔍 나의 외모점수는?", "이미지를 업로드하고 '외모 분석해줘'를 입력해보세요. AI가 외모를 분석해 새로운 매력을 찾아드립니다."),
    ("🎨 웹툰 속으로", "'웹툰화 해줘'라고 입력하면, 사진이 웹툰 주인공처럼 변신합니다."),
    ("📊 이미지 분석", "'이미지 분석해줘'를 입력해 사진 속 숨겨진 정보를 확인해보세요."),
    ("🎵 AI 음악 생성", "나만의 음악이 필요하다면, 이미지를 올리고 '음악 만들어줘'라고 해보세요."),
    ("👗 AI 패션 스타일리스트", "나에게 어울리는 스타일이 궁금하다면, 이미지를 올리고 '패션 추천해줘'를 입력해보세요.")
]

for title, description in features:
    st.sidebar.markdown(f'<p class="feature-title">{title}</p>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<p class="sidebar-text">{description}</p>', unsafe_allow_html=True)


huggingface_api=os.environ["HUGGINGFACE_API_KEY"]
GEMINI_MODEL = 'gemini-1.5-flash'
@st.cache_resource
def load_models():
    genai.configure(api_key='AIzaSyDcq3ZfAUo1i6_24CelEizJftuEkaAPz38')
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    facescore_model = load_model('facescore.h5',compile=False)
    webtoon_model=Generator()
    webtoon_model.load_state_dict(torch.load('./weights/face_paint_512_v2.pt', map_location="cpu"))
    webtoon_model.to('cpu').eval()
    return gemini_model, facescore_model, webtoon_model



mp_face_detection = mp.solutions.face_detection

def detect_and_crop_face(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_np = np.array(image)
        results = face_detection.process(image_np)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image_np.shape
            xmin = int(bbox.xmin * iw)
            ymin = int(bbox.ymin * ih)
            width = int(bbox.width * iw)
            height = int(bbox.height * ih)
            xmax = xmin + width
            ymax = ymin + height
            face = image.crop((xmin, ymin, xmax, ymax))
            return face
        else:
            return None


def generate_chat_response(message, gemini_model):
    response = gemini_model.generate_content(message)
    return response.text

def analyze_image(image, gemini_model):
    prompt = """
    이 이미지에 대해 자세히 분석해주세요. 다음 정보를 포함해주세요:
    1. 이미지에서 보이는 주요 객체나 사람들
    2. 배경이나 장소에 대한 설명
    3. 이미지의 전체적인 분위기나 느낌
    4. 이미지에서 읽을 수 있는 텍스트 (있는 경우)
    5. 이미지의 색상이나 구도에 대한 간단한 설명
    6. 이미지가 전달하려는 메시지나 의미 (있다고 생각되는 경우)
    분석 결과를 한국어로 제공해주세요.
    """
    
    response = gemini_model.generate_content([prompt, image])
    return response.text

def process_facescore(image, facescore_model, gemini_model):
    face = detect_and_crop_face(image)
    if face is None:
        return "얼굴이 감지되지 않았습니다. 다른 이미지를 시도해 주세요."
    
    analysis = analyze_image(image, gemini_model)
    
    face_np = np.array(face)
    img_resized = cv2.resize(face_np, (350, 350))
    img_resized = img_resized.astype(np.float32) / 255.
    img_batch = np.expand_dims(img_resized, axis=0)
    score = facescore_model.predict(img_batch)
    
    if isinstance(score, np.ndarray) and score.size > 1:
        score = score[0]
    score = float(score)
    score = display_result(score)
    
    return f'### 이미지 분석 결과 ###\n\n{analysis}\n\n### 외모점수 결과(1~5) ###\n\n{score}'


def generate_music(image, gemini_model, suno_cookie):
    st.info('3분정도 소요됩니다!')
    face = detect_and_crop_face(image)
    if face is None:
        return "얼굴이 감지되지 않았습니다. 다른 이미지를 시도해 주세요."
    
    prompt = """
    이 이미지에 대해 자세히 분석해주세요. 다음 정보를 포함해주세요:
    1. 성별:
    2. 나이:
    3. 표정:
    분석 결과를 한국어로 간략하게 제공해주세요.
    """
    
    response = gemini_model.generate_content([prompt, image])
    music_path = generate_songs(response.text, suno_cookie)

    # 음악 재생
    st.audio(str(music_path), format='audio/wav')
    st.caption("Generated Music")
    
    # 음악 파일 삭제
    if os.path.exists(music_path):
        os.remove(music_path)

    return "음악이 생성되었습니다. 위의 플레이어에서 음악을 들어보세요."

def generate_songs(result_output, suno_cookie):
    client = suno.Suno(cookie=suno_cookie)
    songs = client.generate(
        prompt=f'{result_output}', is_custom=False, wait_audio=True
    )
    
    # 다운로드할 파일 경로
    file_path = client.download(song=songs[0])
    return file_path
def display_result(score):
    result = round(score, 1)+0.3
    messages = [
        ("'자신감 폭발 중'입니다! 😎 당신은 자신의 외모에 대한 확신으로 가득 차 있어요! %.1f점이라니, 점수와 상관없이 당신의 멋짐은 끝이 없네요! 🤩 당신의 외모는 마치 마법사처럼 사람들을 매료시키고, 누구나 당신을 보면 눈을 뗄 수 없을 거에요! 🪄🧙‍♂️ 비결이 뭐냐고 묻는 사람들에게 자신감이라는 마법의 주문을 알려주세요! 오늘도 당신의 자신감으로 세상을 빛내고, 마법 같은 하루를 보내세요!", 1),
        ("'외모 스승님'입니다. 👩‍🏫 당신의 외모 비결을 전수받고 싶어하는 사람들이 줄을 설 거에요! %.1f점이라는 점수가 무색할 정도로, 당신의 빛나는 외모는 사람들의 눈을 사로잡습니다! ✨ 이제 사람들은 당신의 비밀을 알고 싶어서 질문 세례를 퍼부을 거에요! 외모 스승님으로서 멋지게 대답해 주시고, 사람들에게 당신만의 외모 팁을 살짝 전해 주세요! 다른 사람들은 당신을 닮기 위해 많은 노력을 할 거랍니다!", 1.5),
        ("'외모 아티스트'입니다. 💄 화장품 브랜드들이 당신을 모델로 쓰고 싶어할 만큼 독보적인 매력을 가지고 있네요! %.1f점이라고 해서 당신의 외모가 평범하지 않습니다. 오히려 '매력의 정점'에 도달한 모습이에요! 💃 당신의 멋진 외모를 부러워하는 사람들로 인해 언제나 주목받게 될 거에요! 마치 아티스트처럼 자신만의 스타일을 완성한 당신은 외모계의 진정한 아이콘입니다! 오늘도 당신만의 특별한 매력을 발산하며 하루를 즐기세요!", 2),
        ("외모점수 %.1f점, '미소 전문가'입니다. 😄 당신의 환한 미소는 주변 사람들을 행복하게 만들고, 어디서든 밝은 에너지를 퍼뜨릴 거에요! '미소 기계'라 불리는 당신은 항상 긍정적인 에너지로 가득 차 있답니다! 😁 사람들은 당신의 미소 비결을 배우기 위해 애쓸 거에요! 외모뿐만 아니라 미소로도 사람들의 마음을 사로잡는 당신! 오늘도 환한 미소로 세상을 밝혀주시고, 모두에게 행복을 전해 주세요!", 2.5),
        ("'외모 스타'입니다. 🌟 당신은 거울 속에서 별이 빛나는 모습을 보고도 놀라지 않겠죠! %.1f점이라니, 당신은 외모계의 진정한 스타입니다! 💫 당신의 빛나는 외모와 독특한 스타일은 모두가 부러워하고, 따라가고 싶어할 겁니다! 사람들은 당신을 보고 영감을 받을 거에요! 오늘도 당신만의 특별한 매력으로 주변 사람들을 사로잡고, 당당히 외모계를 이끌어가세요! 당신의 빛나는 외모가 모두에게 희망을 줄 거에요!", 3),
        ("'외모 퀸'입니다. 👸 주변 사람들은 당신의 외모에 주목하고, 귀를 기울일 겁니다! %.1f점이라는 점수가 무색할 정도로, 이제 당신은 외모계의 로열티입니다! 👑 당신의 고급스러운 외모와 독보적인 스타일은 모두가 따라하고 싶어할 거에요! 당신의 외모 비결을 벤치마킹하려는 사람들로 인해 언제나 주목받게 될 겁니다! 여왕처럼 당당히 당신의 외모를 뽐내고, 주변 사람들에게 영감을 주세요! 오늘도 자신감 넘치는 하루 보내세요!", 3.5),
        ("외모점수 %.1f점, '외모의 신화'입니다. 🦄 당신을 보는 사람들은 마치 신화와 전설 속 인물을 보는 듯한 기분을 느낄 겁니다! 외모계의 '뷰티 아카데미 수상자'답게, 당신의 외모는 모두에게 큰 영감을 줄 거에요! 🏆 사람들은 당신의 비결을 배우려고 애쓸 테니, 언제나 자신만의 스타일을 유지하며 그들에게 귀감이 되어주세요! 신화 속 주인공처럼 당신의 외모는 언제나 빛날 겁니다! 오늘도 신화처럼 멋진 하루 보내세요!", 4),
        ("'외모의 황금빛'입니다. 💛 주변에서 당신을 보면 마치 하트가 뿅뿅 튀는 듯한 느낌이 들 거에요! %.1f점이라니, 정말 외모계의 전설답습니다! 🌠 당신의 독보적인 외모와 매력은 누구도 따라올 수 없을 만큼 빛납니다! 다른 사람들이 당신을 따라잡으려면 엄청난 노력이 필요할 거에요! 당신의 황금빛 외모와 매력으로 모두를 사로잡으세요! 오늘도 당신만의 황금빛 미소로 세상을 밝혀주시고, 모두에게 영감을 주세요!", 4.5),
        ("5점 외모, '외모의 신'입니다. 외모계에서 당신을 따라잡으려면 진정한 영웅이 필요할 겁니다! 🦸‍♂️🦸‍♀️ 당신은 외모계의 '뷰티 신'! 🌟 당신의 빛나는 외모와 독보적인 스타일은 모두가 따라하고 싶어할 거에요! 이제 당신은 외모계의 전설이자 영웅입니다! 사람들은 당신을 닮고 싶어하고, 당신의 비결을 배우려고 애쓸 겁니다! 오늘도 외모계의 신으로서 세상을 빛내고, 모두에게 영감을 주세요! 당신의 존재만으로도 세상은 더 밝아질 거에요!", 5)
    ]
    for msg, threshold in messages:
        if result < threshold:
            return msg % result if '%.1f' in msg else msg
@torch.no_grad()
def webtoon(image, webtoon_model, device='cpu'):
    # Move model to the specified device
    webtoon_model = webtoon_model.to(device)
    
    # Resize image if it's too large
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    
    # Convert image to tensor and move to device
    image_tensor = to_tensor(image).unsqueeze(0).to(device) * 2 - 1
    
    # Process the image
    with torch.inference_mode():
        output = webtoon_model(image_tensor, False)
    
    # Move output back to CPU and convert to PIL image
    output = output.cpu().squeeze(0).clip(-1, 1) * 0.5 + 0.5
    output = to_pil_image(output)
    
    return output
def fashion(image, gemini_model,huggingface_api):
    # Gemini API call
    sd_prompt = prompt="""
    Analyze this image in one sentence:
    1. The person visible in the image, including their gender and any notable features
    2. Recommend a new outfit style that would suit the person based on the image
    3. Suggest complementary fashion items or accessories to complete the look
    Present the output in the format:
    "A korean [gender] wearing [recommended outfit and accessories]."
    Example: "A korean woman wearing a white t-shirt and black pants with a bear on it."
    """
    style_prompt="""
    이 이미지에 대해 자세히 분석해주세요. 다음 정보를 포함해주세요:
    1. 이미지에 보이는 주요 인물의 특징 (성별, 나이대, 체형, 얼굴형 등)
    2. 인물이 입고 있는 현재의 옷 스타일과 색상
    3. 인물에게 잘 어울릴 만한 옷 스타일 추천 (예: 캐주얼, 비즈니스 캐주얼, 포멀, 스트릿 패션 등)
    4. 추천하는 옷 스타일에 대한 구체적인 이유와 설명
    5. 추천하는 옷 스타일에 어울리는 색상과 패턴
    6. 이미지의 전체적인 분위기와 어울리는 옷 스타일
    분석 결과를 한국어로 제공해주세요.
    """
    try:
        response = gemini_model.generate_content([sd_prompt, image])
        gemini_description = response.text
        recommanded_response=gemini_model.generate_content([style_prompt,image])
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        return None

    # Hugging Face API call
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {huggingface_api}"}

    try:
        payload = {
            "inputs": 'A highly detailed and photorealistic image of ' + gemini_description,
            "negative_prompt": "cartoonish, low quality, blurry, unrealistic, abstract, fantasy",
            "num_inference_steps": 200,  # 이미지 품질 향상을 위해 단계 수를 늘림
            "guidance_scale": 8.0  # 원하는 스타일을 강화
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        image_bytes = response.content
    except requests.exceptions.RequestException as e:
        print(f"Error in Hugging Face API call: {e}")
        return None

    # Attempt to open the image
    try:
        return Image.open(io.BytesIO(image_bytes)),recommanded_response.text
    except Exception as e:
        print(f"Error opening image: {e}")
        return None


def main():
    st.markdown("<h1 class='vibrant-title'>OmniVerse</h1>", unsafe_allow_html=True)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    gemini_model, facescore_model,  webtoon_model = load_models()

    # Sidebar content
    with st.sidebar:
        st.sidebar.markdown('<p class="sidebar-subtitle">😘 Suno Cookie 설정</p>', unsafe_allow_html=True)
        suno_tab = st.radio("탭 선택", ["Suno Cookie 입력", "Suno Cookie 얻는 방법"])

        if suno_tab == "Suno Cookie 입력":
            suno_cookie = st.text_input("Suno Cookie Key", type="password")
        else:
            st.markdown("""
            ### Suno API Key를 얻는 방법
            """)
            st.markdown("<p class='sidebar-text'><a href='https://github.com/bigdefence/Music-Face'>1. Cookie 얻는 방법</a></br><a href='https://suno.com/'>2. Suno 웹사이트로 이동하기</a></p>",unsafe_allow_html=True)

        st.sidebar.markdown('<p class="sidebar-subtitle">😎 개발자 정보</p>', unsafe_allow_html=True)
        st.markdown("<p class='sidebar-text'>**개발자**: 정강빈</br>**버전**: 2.3.0</p>",unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if "content" in message:
                st.markdown(message['content'])
            if "image" in message:
                st.image(message['image'])

    # Image upload
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.session_state.uploaded_image = image

    # Chat input and processing
    if prompt := st.chat_input('질문을 입력하세요'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        
        with st.chat_message('assistant'):
            if "웹툰화" in prompt.lower() and 'uploaded_image' in st.session_state:
                with st.spinner("웹툰화 진행 중..."):
                    webtoon_image = webtoon(st.session_state.uploaded_image, webtoon_model)
                    st.image(webtoon_image, caption="웹툰화된 이미지", use_column_width=True)
                    response = "이미지를 웹툰 스타일로 변환했습니다. 위의 이미지를 확인해주세요."
            elif "외모" in prompt.lower() and 'uploaded_image' in st.session_state:
                with st.spinner("외모 분석 중..."):
                    response = process_facescore(st.session_state.uploaded_image, facescore_model, gemini_model)
            elif "이미지 분석" in prompt.lower() and 'uploaded_image' in st.session_state:
                with st.spinner("이미지 분석 중..."):
                    response = analyze_image(st.session_state.uploaded_image, gemini_model)
            elif "음악" in prompt.lower() and 'uploaded_image' in st.session_state:
                with st.spinner("음악 생성 중..."):
                    if suno_cookie:
                        response = generate_music(st.session_state.uploaded_image, gemini_model, suno_cookie)
                    else:
                        response = "Suno API Key를 입력해 주세요."
            elif "패션" in prompt.lower() and 'uploaded_image' in st.session_state:
                with st.spinner('패션 추천중...'):
                    fashion_image,recommanded_response= fashion(st.session_state.uploaded_image,gemini_model,huggingface_api)
                    st.image(fashion_image, caption="패션추천 이미지", use_column_width=True)
                    response = "사용자에게 어울리는 패션 추천 이미지를 생성했습니다. 위의 이미지를 확인해주세요.\n"+recommanded_response
            else:
                with st.spinner('답변 생성 중...'):
                    response = generate_chat_response(prompt, gemini_model)
                    
            st.markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()
