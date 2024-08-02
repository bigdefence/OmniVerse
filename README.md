# 🤗 OmniVerse Multi-Modal Korean ChatBot 

## 📝 프로젝트 소개

OmniVerse는 다양한 AI 기술을 통합하여 사용자에게 다양한 기능을 제공하는 멀티모달 한국어 챗봇입니다. 이 프로젝트는 텍스트 생성, 이미지 분석, 음악 생성 및 패션 추천을 포함한 여러 기능을 제공합니다.

## ✨ 주요 기능

- 💬 Gemini 챗봇: 다양한 질문에 답변하고 유용한 정보를 제공합니다. 특정 주제에 대한 질문도 가능합니다.
- 🔍 외모 분석: 이미지를 업로드하고 '외모 분석해줘'를 입력하면 AI가 외모를 분석해 새로운 매력을 찾아드립니다.
- 🎨 웹툰화: '웹툰화 해줘'라고 입력하면, 사진이 웹툰 주인공처럼 변신합니다.
- 📊 이미지 분석: '이미지 분석해줘'를 입력해 사진 속 숨겨진 정보를 확인해보세요.
- 🎵 음악 생성: 이미지를 올리고 '음악 만들어줘'라고 하면 나만의 음악을 생성합니다.
- 👗 패션 추천: 이미지를 올리고 '패션 추천해줘'를 입력하면 사용자에게 어울리는 스타일을 추천합니다.

## 🛠️ 기술 스택

- Streamlit
- Torch
- TensorFlow
- Google Generative AI
- MediaPipe
- Hugging Face
- Suno
- Stable Diffusion XL 1.0-base

## 📋 설치 및 실행 방법
### 사전 요구 사항
- Python 3.8 이상
- 필요한 패키지들은 requirements.txt 파일을 참조하여 설치하세요.
1. 저장소 클론:
   ```
   git clone https://github.com/bigdefence/OmniVerse.git
   cd OmniVerse
   ```

2. 필요한 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

3. 각 서비스 실행:
   - 이미지 생성 및 얼굴 분석 서버:
     ```
     fastapi_fashion.ipynb
     ```
   - Streamlit 앱 실행:
     ```
     streamlit run app.py
     ```

## 📄 Suno API Key 설정

- [Suno 웹사이트](https://suno.com/)에서 회원가입 및 API Key를 생성합니다.
- 앱의 사이드바에서 Suno API Key를 입력합니다.

## 👨‍💻 개발자 정보

- **개발자**: 정강빈
- **버전**: 2.0.0

## 🔗 API 엔드포인트

- 이미지 생성 API: `이미지 토큰/fashion`

## 📜 라이선스

이 프로젝트는 [라이선스 이름] 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

---

❗ **참고**: API 엔드포인트는 개발 환경에 따라 변경될 수 있습니다. 실제 배포 시 안정적인 URL로 업데이트해주세요.
