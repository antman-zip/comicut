import uuid
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# --- Data Models ---
class Character(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    image: Optional[str] = None  # Base64 encoded image
    description: str = ""  # New: User's text description (e.g., "Blue suit, short black hair")
    enabled: bool = True

ImageResolution = Literal['1K', '2K', '4K']

# --- New Models for Report Pipeline ---

class FactBank(BaseModel):
    title: str
    summary: str
    key_entities: List[str]
    timeline: List[Dict[str, Any]] = [] # Flexible dict for timeline events

class PipelineState(BaseModel):
    """Tracks the progress of the report-to-comic conversion."""
    report_text: Optional[str] = None
    report_path: Optional[str] = None # Path to the uploaded PDF file
    fact_data: Optional[Dict[str, Any]] = None # Changed to generic dict for flexibility with user code
    draft_scenario: Optional[str] = None
    final_scenario: Optional[str] = None

class ScriptCut(BaseModel):
    cutNumber: int
    description: str

class GeneratedPanel(BaseModel):
    id: str
    cutNumber: int
    description: str
    imageUrl: Optional[str] = None
    status: Literal['pending', 'generating', 'completed', 'error']
    timestamp: float # Assuming timestamp is a float (e.g., from time.time())

class LogEntry(BaseModel):
    id: str
    message: str
    timestamp: datetime
    type: Literal['info', 'success', 'error']
