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

# --- Helper: Image Analysis ---
async def analyze_character_visual(image_bytes: bytes) -> str:
    """Analyzes a character image and returns short visual keywords."""
    # Using the latest fast model for vision tasks
    model = genai.GenerativeModel("gemini-3-flash-preview")
    
    prompt = """
    Analyze this character image and provide a visual description in short, comma-separated keywords (Korean).
    Focus on: Hair style/color, Eye shape/color, Clothing details, Accessories, and distinct features.
    Example: 검은 단발머리, 파란 정장, 빨간 넥타이, 안경 착용, 차분한 인상
    Output ONLY the keywords.
    """
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        response = await model.generate_content_async([prompt, img])
        return response.text.strip()
    except Exception as e:
        print(f"Image analysis error: {e}")
        return "이미지 분석 실패"

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
    validation_prompt = load_user_prompt("verification_prompt_ko.md")
    
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
    Analyze the following comic script and break it down into individual cuts (panels). 
    Return a JSON array where each object contains:
    - cutNumber (int)
    - description (string): A highly detailed visual description for an AI image generator.

    [COMPOSITION & DIRECTING RULES FOR 'description']
    1. **Static/Calm**: DO NOT describe dynamic actions (running, jumping, fighting). Keep poses static or subtle.
    2. **Characters**: Limit to 1 or 2 characters max per cut.
    3. **Assets/Objects**: If the cut describes an abstract 'Image/Asset' (logo, chart, chip, product), describe it EXACTLY as: "A [Object Name] placed in the center of a Solid Background." Do not describe a complex background for these cuts.
    
    [TEXT & DIALOGUE RULES]
    1. **NO On-Screen Text**: Do NOT include instructions for subtitles, labels, or titles.
    2. **Speech Bubbles**: If there is dialogue, explicitly write: "Includes a speech bubble with the text: '[Dialogue Content]'." 
    3. **Split Long Dialogue**: If a dialogue line is very long (>50 chars), describe it as "multiple speech bubbles".
    4. **Clean Content**: Ensure the dialogue text in the description is exact.

    [NEGATIVE RULES]
    1. **Remove Metadata**: STRIP OUT all references like "Ref 123", "Fact 1", "Source:", "Page 2". The description must contain NO meta-data.
    
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

    # 1. Construct Character Context (Narrative Style)
    enabled_chars = [c for c in characters if c.enabled]
    char_descriptions = []
    
    for c in enabled_chars:
        desc = f"Name: {c.name}"
        if c.description:
            desc += f", Appearance: {c.description}"
        char_descriptions.append(desc)
    
    char_block = "\n".join(char_descriptions)

    # 2. Build Structured Prompt
    full_prompt = f"""
    한국 웹툰 스타일의 고품질 컷(1컷) 이미지를 생성하세요. (1:1 비율 준수)

    [스타일 가이드 / Style Guide]
    1. **캐릭터 (Character Style)**:
       - **Anime Coloring, Flat Color**: 평면적이고 깔끔한 셀 채색(Cel-shading) 스타일.
       - 과도한 그라데이션이나 3D 느낌을 배제하고, 선명한 라인과 단색 위주의 채색 사용.
    
    2. **배경 및 에셋 (Background & Asset Style)**:
       - **Realistic Learning-Manga Style Anime**: 실사에 가까운 디테일한 학습만화풍 애니메이션 스타일.
       - 배경이나 사물(에셋)은 캐릭터보다 더 사실적이고 밀도 있게 묘사하여 깊이감을 줄 것.

    [전체 톤앤매너]
    - 차분한 분위기, 과장된 코믹 연출 금지.
    - 색감은 저채도 위주로 안정감 있게.
    - 해상도는 1:1 비율(Square) 필수.

    [캐릭터 정보]
    {char_block}
    - **의상 고정 (CRITICAL OUTFIT ENFORCEMENT)**:
      1. 위 [캐릭터 정보]에 적힌 외관(의상, 헤어 등)이 **절대적인 우선순위**를 가짐.
      2. 아래 [장면 설명]에 '정장(suit)', '유니폼' 등 다른 의상 묘사가 나오더라도 **전부 무시(IGNORE)**하고, 반드시 위 캐릭터 정보의 의상을 입힐 것.
      3. 캐릭터의 고유 외관을 유지하는 것이 가장 중요함.

    [장면 설명]
    {prompt}
    """
    
    # 3. Add Reference Images
    content_parts: list = [full_prompt]

    for char in enabled_chars:
        if char.image:
            try:
                # Handle data URI or base64 raw
                img_data_str = char.image
                if "," in img_data_str:
                    header, base64_data = img_data_str.split(',', 1)
                else:
                    base64_data = img_data_str
                
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
