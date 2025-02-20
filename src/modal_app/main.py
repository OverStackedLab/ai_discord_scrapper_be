# src/modal_app/main.py
import sqlite3
from modal import asgi_app
from .common import DB_PATH, VOLUME_DIR, app, fastapi_app, volume


@app.function(
    volumes={VOLUME_DIR: volume},
)
def init_db():
    """Initialize the SQLite database with a simple table."""
    volume.reload()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create a simple table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()
    volume.commit()


@app.function(
    volumes={VOLUME_DIR: volume},
)
@asgi_app()
def fastapi_entrypoint():
    # Initialize database on startup
    init_db.remote()
    return fastapi_app


@fastapi_app.post("/items/{name}")
async def create_item(name: str):
    volume.reload()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO items (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()
    volume.commit()
    return {"message": f"Added item: {name}"}


@fastapi_app.get("/items")
async def list_items():
    volume.reload()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM items")
    items = cursor.fetchall()
    conn.close()
    return {
        "items": [
            {"id": item[0], "name": item[1], "created_at": item[2]} for item in items
        ]
    }


@fastapi_app.get("/")
def read_root():
    return {"message": "Hello World"}
