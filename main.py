from dotenv import load_dotenv
load_dotenv()

import uvicorn
from app import app

if __name__ == "__main__":
    """
    This is the main entry point to run the FastAPI application.
    
    To run this server, execute the following command in your terminal
    from inside this directory (C:\AI\comicut-ai\python_app):
    
    uvicorn main:app --reload
    
    You will need to have uvicorn and other dependencies installed:
    pip install -r requirements.txt
    """
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
