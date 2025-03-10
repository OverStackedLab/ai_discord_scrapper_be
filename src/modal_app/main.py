import os
import sqlite3
from datetime import datetime

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from modal import asgi_app
from openai import OpenAI
from pydantic import BaseModel
from typing_extensions import Union

from .agent import process_agent_message
from .common import DB_PATH, VOLUME_DIR, app, fastapi_app, volume


class TokenData(BaseModel):
    access_token: str


class AgentRequest(BaseModel):
    message: str


class AgentResponse(BaseModel):
    response: str


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
            CREATE TABLE IF NOT EXISTS google_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                token_expiry TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
    )
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS agent_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


@fastapi_app.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(request: AgentRequest):
    volume.reload()
    try:
        result = process_agent_message(request.message)
        return {"response": result}
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.post("/auth/google/token")
def receive_token(token_data: TokenData):
    try:
        # Create credentials using the provided access token. this call will fail if we don't have the proper credentials
        creds = Credentials(
            token_data.access_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        )
        # Extract additional token details if available.
        refresh_token = creds.refresh_token if creds.refresh_token else ""
        token_expiry = creds.expiry.isoformat() if creds.expiry else ""
        # For simplicity, remove any previously stored token and insert the new one.
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM google_tokens")
        cursor.execute(
            "INSERT INTO google_tokens (access_token, refresh_token, token_expiry) VALUES (?, ?, ?)",
            (token_data.access_token, refresh_token, token_expiry),
        )
        conn.commit()
        conn.close()
        volume.commit()

        # Return the token info along with the test events.
        return {
            "access_token": token_data.access_token,
            "refresh_token": refresh_token,
            "token_expiry": token_expiry,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@fastapi_app.delete("/agent/thread")
def delete_agent_thread():
    """
    Delete the stored agent thread from the SQLite database.
    This will force the next agent request to create a new thread.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM agent_threads")
        conn.commit()
        conn.close()
        volume.commit()
        return {"message": "Agent thread deleted successfully."}
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@fastapi_app.get("/agent/history")
def get_agent_history():
    """
    Retrieve the entire chat history for the current agent thread.
    """
    try:
        # Retrieve the current thread ID from the agent_threads table.
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT thread_id FROM agent_threads ORDER BY updated_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {"messages": []}
        thread_id = row[0]

        # Use the OpenAI Assistants API to list messages from the thread.
        messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        chat_history = []
        if messages.data:
            for m in messages.data:
                # Assume that each message contains at least one text element.
                role = m.role
                # Adjust this line based on your SDK's response structure.
                text = (
                    m.content[0].text.value
                    if m.content and m.content[0].text.value
                    else ""
                )
                chat_history.append({"role": role, "text": text})
        return {"messages": chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.get("/")
def read_root():
    return {"message": "Hello World"}
