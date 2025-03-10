import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modal import App, Image, Secret, Volume

DB_FILENAME = "google_agent.db"
VOLUME_DIR = "/cache-vol"
DB_PATH = pathlib.Path(VOLUME_DIR, DB_FILENAME)


volume = Volume.from_name("sqlite-db-vol", create_if_missing=True)
image = Image.debian_slim().pip_install_from_pyproject("pyproject.toml")
secrets = Secret.from_dotenv()

app = App(name="starter_template", secrets=[secrets], image=image)

# Create a FastAPI instance here so it can be shared across modules
fastapi_app = FastAPI()

# Configure CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
