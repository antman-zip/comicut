import os
import json
import base64
import io
from typing import List, Dict, Any

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from PIL import Image

# Import models
from models import Character, ImageResolution, ScriptCut, FactBank

# Try to import user ingestion logic
try:
    from user.factbank_ingest import extract_facts_from_pdf
except ImportError:
    print("Warning: Could not import 'user.factbank_ingest.extract_facts_from_pdf'. PDF extraction may fail.")
    def extract_facts_from_pdf(pdf_path):
        return {"error": "Module not found"}

# --- Configuration ---
def configure_genai():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY or GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    return api_key

API_KEY = configure_genai()

# --- Constants ---
PARSE_MODEL = "gemini-3-pro-preview"
WRITER_MODEL = "gemini-3-pro-preview"
IMAGE_GEN_MODEL = "gemini-3-pro-image-preview"

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- Helper: Load Prompts ---
def load_user_prompt(filename: str) -> str:
    """Loads a markdown prompt file from the user/ directory."""
    path = os.path.join("user", filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: Prompt file '{filename}' not found."

# --- Pipeline Step 1: Ingest Report (Uses user logic) ---
async def process_report_to_facts(report_source: str, is_file_path: bool = False) -> Dict[str, Any]:
    """
    Step 1: Convert raw report (PDF path or Text) into a FactBank JSON structure.
    Delegates to the user-defined logic in `user/factbank_ingest.py`.
    """
    if is_file_path:
        # Assuming report_source is a path to a PDF file
        print(f"Processing PDF from path: {report_source}")
        return extract_facts_from_pdf(report_source)
    else:
        # Legacy/Text fallback - if the user code supported text directly we'd call it.
        # But the current user code is strictly PDF file based.
        # We can simulate a minimal FactBank for raw text if needed, or raise error.
        return {
            "title": "Text Input",
            "facts": [{"claim": report_source, "tags": ["raw_text"]}],
            "error": "Raw text input is not fully supported by the high-quality PDF ingestor."
        }

# --- Pipeline Step 2: Draft Scenario ---
async def generate_scenario_draft(fact_data: Dict[str, Any]) -> str:
    """Step 2: Generate a draft scenario from FactBank JSON using the Writer System Prompt."""
    model = genai.GenerativeModel(WRITER_MODEL)
    system_prompt = load_user_prompt("scenario_writer_system_prompt_ko.md")
    
    # Convert JSON back to string for the prompt
    fact_str = json.dumps(fact_data, ensure_ascii=False, indent=2)
    
    full_prompt = f"""
    {system_prompt}

    [FactBank Data]
    {fact_str}
    """
    
    response = await model.generate_content_async(full_prompt, safety_settings=SAFETY_SETTINGS)
    return response.text

# --- Pipeline Step 3: Validate & Refine ---
async def refine_scenario(draft_scenario: str) -> str:
    """Step 3: Refine the draft scenario for image generation using the Validation Prompt."""
    model = genai.GenerativeModel(WRITER_MODEL)
    validation_prompt = load_user_prompt("검증프롬프트_ko.md")
    
    full_prompt = f"""
    {validation_prompt}

    [시나리오 초안]
    {draft_scenario}
    """
    
    response = await model.generate_content_async(full_prompt, safety_settings=SAFETY_SETTINGS)
    return response.text

# --- Pipeline Step 4: Parse Script (Existing but updated) ---
async def parse_script(script_text: str) -> List[ScriptCut]:
    """Step 4: Parse the FINAL refined script into structured cuts for the UI/Image Gen."""
    model = genai.GenerativeModel(PARSE_MODEL)
    
    prompt = f"""
    Analyze the following comic script. Break it down into individual cuts (panels). 
    Return a JSON array where each object contains:
    - cutNumber (int)
    - description (string): The COMPLETE visual description for the image generator. 
      WARNING: Do not truncate or summarize the description. Include all details, character actions, and dialogue instructions exactly as they appear in the script.
    
    The script is already formatted for image generation. Just extract the fields. 
    
    Script:
    {script_text}
    """

    response = await model.generate_content_async(
        prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            max_output_tokens=15000
        ),
        safety_settings=SAFETY_SETTINGS
    )
    
    try:
        data = json.loads(response.text)
        return [ScriptCut(**item) for item in data]
    except Exception as e:
        print(f"Error parsing script JSON: {e}")
        # Fallback: Try to find JSON block if the model was chatty
        # (Simplified logic for now)
        raise ValueError("Failed to parse script structure.") from e

# --- Pipeline Step 5: Generate Image (Existing) ---
async def generate_panel_image(
    prompt: str,
    characters: List[Character],
    resolution: ImageResolution,
    global_context: str = "",
    prev_desc: str = None,
    next_desc: str = None
) -> tuple[str, str]:
    """Step 5: Generate the image for a specific panel."""
    model = genai.GenerativeModel(IMAGE_GEN_MODEL)

    # User's Custom Korean Prompt + 1:1 Aspect Ratio Enforcement
    full_prompt = f"""한국 웹툰 스타일의 고품질 컷(1컷) 이미지를 생성하세요. (1:1 비율 준수)

[스타일/톤]
- 차분한 분위기, 과장된 코믹 연출 금지
- 깔끔한 선화, 적당한 채색, 과한 광원/네온/현란한 배경 금지
- 배경에 이미지/에셋에 해당하는 배경 혹은 오브젝트 배치
- 색감은 저채도(파스텔/뉴트럴) 위주, 대비 과하지 않게
- 해상도는 1:1 비율(Square)로 출력.

[구도/연출]
- 역동적인 액션(달리기/점프/격투) 금지
- 장면은 단일 인물 또는 2인까지, 화면이 복잡해지지 않게
- **중요**: 시나리오에 '이미지/에셋'(예: 로고, 칩, 제품, 특정 사물)이 묘사된 컷은 배경 묘사를 생략하고 반드시 "단색 배경(Solid Background) 중앙에 배치된 [사물]" 형태로 묘사하여 깔끔하게 출력되도록 할 것.

[텍스트 규칙 — 매우 중요]
- 온스크린 텍스트(자막/설명문/수치 텍스트/제목/라벨) 생성 금지
- 예외는 오직 "말풍선" 안의 대사 1개(또는 대화 2개)만 허용, 50자 넘어가는 대사는 문장마다 말풍선 하나씩 추가.
- 말풍선은 반드시 포함 (speech bubble 필수)
- **핵심**: 말풍선 안에 대사를 가능한 정확하게 기입할 것.

[장면 설명]
{prompt}"""
    
    enabled_chars = [c for c in characters if c.enabled and c.image]
    if enabled_chars:
        char_names = ", ".join([c.name for c in enabled_chars])
        full_prompt += f"""

[캐릭터 일관성 유지 - 필수]
- 등장 인물: {char_names}
- **참조 이미지 준수**: 제공된 캐릭터 참조 이미지의 시각적 디테일(헤어스타일, 색상, 눈 모양, 의상 등)을 **정확히** 따를 것.
- 장면 설명에 의상 변경 지시가 없다면 참조 이미지의 의상을 그대로 유지할 것."""

    content_parts: list = [full_prompt]

    for char in enabled_chars:
        if char.image:
            try:
                header, base64_data = char.image.split(',', 1)
                image_data = base64.b64decode(base64_data)
                img = Image.open(io.BytesIO(image_data))
                content_parts.append(img)
            except Exception as e:
                print(f"Skipping character image {char.name}: {e}")

    try:
        response = await model.generate_content_async(content_parts, safety_settings=SAFETY_SETTINGS)
        
        b64_str = None
        # Check output structure (Inline data)
        if response.parts and hasattr(response.parts[0], 'inline_data'):
             d = response.parts[0].inline_data
             b64_str = base64.b64encode(d.data).decode('utf-8')
             return f"data:{d.mime_type};base64,{b64_str}", full_prompt
        
        # Check candidates fallback
        if response.candidates and response.candidates[0].content.parts:
             p = response.candidates[0].content.parts[0]
             if hasattr(p, 'inline_data'):
                 b64_str = base64.b64encode(p.inline_data.data).decode('utf-8')
                 return f"data:{p.inline_data.mime_type};base64,{b64_str}", full_prompt

        raise ValueError("No image data returned.")

    except Exception as e:
        print(f"Image gen error: {e}")
        raise
