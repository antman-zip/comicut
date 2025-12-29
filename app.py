import asyncio
import base64
import io
import logging
import uuid
import zipfile
import json
import os
import shutil
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import core types and services
from models import (
    Character,
    GeneratedPanel,
    ImageResolution,
    LogEntry,
    ScriptCut,
    FactBank, 
    PipelineState
)
from services.gemini_service import (
    process_report_to_facts,
    generate_scenario_draft,
    refine_scenario,
    parse_script,
    generate_panel_image,
)
import services.gemini_service as gemini_service

# --- Application Setup ---
app = FastAPI(
    title="ComiCut AI",
    description="Convert reports to comic panels using AI pipeline.",
    version="2.0.0",
)

# --- Static Files Setup ---
IMG_DIR = os.path.join(os.getcwd(), "generated_images")
os.makedirs(IMG_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=IMG_DIR), name="static")

# Serve Favicon/Icon from root
@app.get("/icon.png", include_in_schema=False)
async def get_icon():
    icon_path = "icon.png"
    if os.path.exists(icon_path):
        return FileResponse(icon_path)
    raise HTTPException(status_code=404, detail="Icon not found")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- In-Memory State Management ---
class AppState:
    def __init__(self):
        self.pipeline = PipelineState()
        self.file_name: Optional[str] = None
        self.characters: Dict[str, Character] = {}
        self.resolution: ImageResolution = "1K"
        self.logs: List[LogEntry] = []
        self.panels: Dict[str, GeneratedPanel] = {}
        self.stop_signal = asyncio.Event()
        self.is_generating = False 

    def add_log(self, message: str, type: str = "info"):
        log_entry = LogEntry(
            id=str(uuid.uuid4()),
            message=message,
            timestamp=datetime.now(),
            type=type,
        )
        self.logs.append(log_entry)
        logging.info(f"[{type.upper()}] {message}")

    def reset_for_generation(self):
        self.panels.clear()
        self.stop_signal.clear()
        self.is_generating = False
        
        # Clear old images
        for f in os.listdir(IMG_DIR):
            try:
                os.remove(os.path.join(IMG_DIR, f))
            except Exception as e:
                print(f"Error clearing old image {f}: {e}")
            
        self.add_log("Previous image generation cleared. Ready for new run.")

state = AppState()


# --- Pydantic Models ---
class CharacterUpdateRequest(BaseModel):
    name: str
    image: Optional[str] = None
    description: Optional[str] = ""
    enabled: bool

class GenerationRequest(BaseModel):
    script_text: Optional[str] = None

class RegenerateRequest(BaseModel):
    panel_id: str
    new_description: str

class UpdateFactRequest(BaseModel):
    facts: Dict[str, Any]

class UpdateScenarioRequest(BaseModel):
    scenario_text: str

# --- API Endpoints ---

@app.get("/", summary="Check server status.")
def read_root():
    return {"message": "ComiCut AI (Pipeline Version) is running."}

# 1. Upload
@app.post("/upload-report/", summary="Upload a report file (Step 1).")
async def upload_report_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")
    try:
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        safe_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(tmp_dir, safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        state.pipeline = PipelineState()
        state.pipeline.report_path = file_path
        state.file_name = file.filename
        state.pipeline.report_text = f"File saved at: {file_path}"
        state.add_log(f"File uploaded and saved: {file.filename}", "success")
        return {"filename": file.filename, "path": file_path}
    except Exception as e:
        state.add_log(f"Failed to save file: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

# 2. Facts
@app.post("/process/facts/", summary="Extract facts (Step 2).")
async def process_facts():
    if state.pipeline.report_path:
        source = state.pipeline.report_path
        is_path = True
    elif state.pipeline.report_text:
        source = state.pipeline.report_text
        is_path = False
    else:
        raise HTTPException(status_code=400, detail="No report loaded.")
    state.add_log(f"Extracting facts from {'PDF file' if is_path else 'text'}")
    try:
        facts_dict = await process_report_to_facts(source, is_file_path=is_path)
        state.pipeline.fact_data = facts_dict
        state.add_log("Fact extraction complete.", "success")
        return state.pipeline.fact_data
    except Exception as e:
        state.add_log(f"Fact extraction failed: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/facts/")
def get_facts(): return state.pipeline.fact_data

@app.put("/pipeline/facts/")
def update_facts(payload: UpdateFactRequest):
    state.pipeline.fact_data = payload.facts
    state.add_log("Facts updated manually.")
    return {"message": "Facts updated."}

# 3. Draft
@app.post("/process/draft/", summary="Draft scenario (Step 3).")
async def process_draft():
    if not state.pipeline.fact_data:
        raise HTTPException(status_code=400, detail="No fact data available.")
    state.add_log("Generating scenario draft...")
    try:
        draft = await generate_scenario_draft(state.pipeline.fact_data)
        state.pipeline.draft_scenario = draft
        state.add_log("Draft scenario generated.", "success")
        return {"draft": draft}
    except Exception as e:
        state.add_log(f"Draft generation failed: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/pipeline/draft/")
def update_draft(payload: UpdateScenarioRequest):
    state.pipeline.draft_scenario = payload.scenario_text
    state.add_log("Draft scenario updated manually.")
    return {"message": "Draft updated."}

# 4. Refine
@app.post("/process/refine/", summary="Refine scenario (Step 4).")
async def process_refine():
    if not state.pipeline.draft_scenario:
        raise HTTPException(status_code=400, detail="No draft scenario available.")
    state.add_log("Refining scenario for image generation...")
    try:
        final = await refine_scenario(state.pipeline.draft_scenario)
        state.pipeline.final_scenario = final
        state.add_log("Scenario refined and validated.", "success")
        return {"final_scenario": final}
    except Exception as e:
        state.add_log(f"Refinement failed: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/pipeline/final/")
def update_final(payload: UpdateScenarioRequest):
    state.pipeline.final_scenario = payload.scenario_text
    state.add_log("Final scenario updated manually.")
    return {"message": "Final scenario updated."}

# 5. Generate
@app.post("/generate/start/", summary="Start generation.")
async def start_generation(background_tasks: BackgroundTasks, request: Optional[GenerationRequest] = None):
    if state.is_generating:
        raise HTTPException(status_code=409, detail="A generation process is already running.")
    script_source = None
    if request and request.script_text:
        script_source = request.script_text
    else:
        script_source = state.pipeline.final_scenario or state.pipeline.draft_scenario
    if not script_source:
         raise HTTPException(status_code=400, detail="No scenario found.")
    state.reset_for_generation()
    state.is_generating = True
    state.add_log("Starting image generation process...")
    background_tasks.add_task(_generation_workflow, script_source)
    return {"message": "Generation process started."}

async def _generation_workflow(script_text: str):
    try:
        state.add_log("Analyzing final script to identify cuts...")
        cuts = await parse_script(script_text)
        state.add_log(f"Analysis complete: {len(cuts)} cuts found.", "success")
        if state.stop_signal.is_set(): raise InterruptedError("Stopped.")
        
        new_panels = [
            GeneratedPanel(
                id=str(uuid.uuid4()),
                cutNumber=cut.cutNumber,
                description=cut.description,
                status='pending',
                timestamp=datetime.now().timestamp(),
            )
            for cut in cuts
        ]
        for p in new_panels: state.panels[p.id] = p
        
        # Global Context
        global_context = state.pipeline.final_scenario or script_text
        
        for i, panel in enumerate(new_panels):
            if state.stop_signal.is_set():
                state.add_log("Generation stopped.", "info")
                break
            
            # Local Context
            prev_desc = new_panels[i-1].description if i > 0 else None
            next_desc = new_panels[i+1].description if i < len(new_panels) - 1 else None
            
            await _generate_single_panel_task(
                panel.id, 
                panel.description, 
                global_context=global_context,
                prev_desc=prev_desc,
                next_desc=next_desc
            )
        
        if not state.stop_signal.is_set():
            state.add_log("All cuts have been generated.", "success")
    except Exception as e:
        state.add_log(f"Generation failed: {str(e)}", "error")
        print(traceback.format_exc())
    finally:
        state.is_generating = False 

@app.get("/logs/", response_model=List[LogEntry])
def get_logs(): return state.logs

@app.get("/panels/", response_model=List[GeneratedPanel])
def get_panels():
    return sorted(list(state.panels.values()), key=lambda p: p.cutNumber)

@app.post("/generate/stop/")
async def stop_generation():
    if not state.is_generating: raise HTTPException(status_code=400, detail="Not running.")
    state.stop_signal.set()
    state.add_log("Stop signal received.")
    return {"message": "Stopping..."}

@app.post("/regenerate/")
async def regenerate_panel(request: RegenerateRequest):
    if state.is_generating: raise HTTPException(status_code=409, detail="Busy.")
    panel = state.panels.get(request.panel_id)
    if not panel: raise HTTPException(status_code=404)
    await _generate_single_panel_task(request.panel_id, request.new_description, global_context=state.pipeline.final_scenario)
    return state.panels.get(request.panel_id)

async def _save_image_to_file(base64_string: str, panel_id: str) -> str:
    try:
        header, base64_data = base64_string.split(',', 1)
        file_ext = "png"
        if "jpeg" in header or "jpg" in header:
            file_ext = "jpeg"
        elif "webp" in header:
            file_ext = "webp"
        
        filename = f"{panel_id}.{file_ext}"
        file_path = os.path.join(IMG_DIR, filename)
        
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
        return file_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return base64_string

async def _generate_single_panel_task(
    panel_id: str, 
    description: str,
    global_context: str = "",
    prev_desc: str = None,
    next_desc: str = None
):
    panel = state.panels.get(panel_id)
    if not panel: return
    panel.status = "generating"
    panel.description = description
    try:
        state.add_log(f"Generating image for Cut #{panel.cutNumber}")
        enabled_chars = [c for c in state.characters.values() if c.enabled]
        
        base64_image, used_prompt = await generate_panel_image(
            description, 
            enabled_chars, 
            state.resolution,
            global_context=global_context,
            prev_desc=prev_desc,
            next_desc=next_desc
        )
        
        state.add_log(f"[PROMPT Cut #{panel.cutNumber}]\n{used_prompt}", "info")
        
        image_url = await _save_image_to_file(base64_image, panel.id)
        panel.imageUrl = image_url
        panel.status = "completed"
        state.add_log(f"Cut #{panel.cutNumber} ready.", "success")
    except Exception as e:
        panel.status = "error"
        state.add_log(f"Failed Cut #{panel.cutNumber}: {e}", "error")

# Characters
@app.post("/characters/", response_model=Character)
def add_character():
    if len(state.characters) >= 3: raise HTTPException(status_code=400)
    char_id = str(uuid.uuid4())
    new_char = Character(id=char_id, name="", image=None, enabled=True)
    state.characters[char_id] = new_char
    state.add_log(f"Added character slot {char_id}")
    return new_char

@app.put("/characters/{char_id}", response_model=Character)
def update_character(char_id: str, payload: CharacterUpdateRequest):
    if char_id not in state.characters: raise HTTPException(status_code=404)
    char = state.characters[char_id]
    char.name = payload.name
    char.image = payload.image
    char.description = payload.description
    char.enabled = payload.enabled
    return char

@app.delete("/characters/{char_id}")
def remove_character(char_id: str):
    if char_id in state.characters: del state.characters[char_id]
    return {"message": "Removed"}

@app.get("/characters/", response_model=List[Character])
def get_characters(): return list(state.characters.values())

# --- ZIP Download & Utilities ---
def create_images_zip():
    """Creates a zip file in the tmp directory and returns its path."""
    completed_panels = [p for p in state.panels.values() if p.status == "completed" and p.imageUrl]
    if not completed_panels:
        raise ValueError("No images to zip.")

    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    zip_filename = f"comicut_{int(datetime.now().timestamp())}.zip"
    zip_path = os.path.join(tmp_dir, zip_filename)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for panel in completed_panels:
            if panel.imageUrl and os.path.exists(panel.imageUrl):
                filename = os.path.basename(panel.imageUrl)
                ext = filename.split('.')[-1]
                zf.write(panel.imageUrl, arcname=f"cut_{panel.cutNumber}.{ext}")
    return zip_path

@app.post("/create-zip/")
async def create_zip_endpoint():
    try:
        zip_path = create_images_zip()
        return {"path": zip_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image/")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """Analyzes an uploaded image and returns visual keywords."""
    try:
        contents = await file.read()
        description = await gemini_service.analyze_character_visual(contents)
        return {"description": description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Mount Gradio UI ---
from gradio import mount_gradio_app
from ui import create_ui

ui = create_ui()
app = mount_gradio_app(app, ui, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
