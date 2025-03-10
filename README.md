# ASR Automatic Speech Recognition

- Running the project

```
$ uv run modal serve src.modal_app.main
```

# Building an Executive Assistant Agents

Today, we’re combining everything we've learned in the course so far by building an **AI agent**—a virtual executive assistant that not only chats with you but can also schedule meetings, send emails, and interact with your Google Calendar.

In this post, we’ll walk through a complete code example of our AI agent and explain each major component step by step.

As always for reference you can checkout the `agent` [branch](https://github.com/nhein-tt/starter_template/tree/agent) from the starter_template up on github.

Let’s dive in!

---

## Prerequisites

Before we get into the code for today, we've got some additonal vendor set up to do. Since we're using the google API's, we'll need to set up with a google developer account. If you've ever used AWS/Azure/GCP, you will know that it is not an easy place to navigate. So instead of telling you to go to X tab on the left and go through the flow, it's going to be much easier in my opinion to watch a short video instead. So just follow along with the short video below to get your google account set up with all of the credentials, authentications, and Oauth flows that you need.

## https://www.youtube.com/watch?v=84p3XzaZSMM

## Overview of the AI Agent

Our AI agent is designed to act as a virtual executive assistant (EA) that helps with everyday tasks such as:

- **Chatting:** Process natural language queries using an LLM.
- **Scheduling Meetings:** Create calendar events via Google Calendar.
- **Sending Emails:** Compose and dispatch emails through Gmail.
- **Managing Conversations:** Persist and retrieve chat history using SQLite.

This functionality is powered by advanced techniques including:

- **Function Calling:** Using a defined set of “tools” (e.g., `schedule_meeting`, `send_email`) to delegate tasks directly to code.
- **Persistent Threads:** Storing conversation threads and Google tokens in a persistent SQLite database attached to a Modal volume.

---

## 1. The Backend: Setting Up the Agent in `main.py`

Our main backend file, `main.py`, ([complete file for reference](https://github.com/nhein-tt/starter_template/blob/agent/backend_service/src/modal_app/main.py)) sets up the Modal functions, database initialization, and API endpoints for our agent. Let’s break down the key parts.

### Initializing the Database

Before any requests are handled, we need to set up our SQLite database. Notice that we’re creating two tables: one for storing Google tokens and one for persisting agent conversation threads.

```python
@app.function(
    volumes={VOLUME_DIR: volume},
)
def init_db():
    """Initialize the SQLite database with a simple table."""
    volume.reload()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create a table to store Google tokens.
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
    # Create a table to store agent conversation threads.
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
```

> **Explanation:**
>
> - Two tables are created:
>   - `google_tokens` stores OAuth credentials.
>   - `agent_threads` persists a conversation thread identifier for our AI agent.

### Exposing Our FastAPI Application

Next, we define our main entrypoint for FastAPI. This function calls `init_db` on startup to ensure our tables exist.

```python
@app.function(
    volumes={VOLUME_DIR: volume},
)
@asgi_app()
def fastapi_entrypoint():
    # Initialize database on startup
    init_db.remote()
    return fastapi_app
```

### API Endpoints for the AI Agent

We expose several endpoints to interact with our agent:

#### 1. Chat Endpoint

This endpoint accepts a user’s message and returns the agent’s response. It delegates processing to our agent logic (in `agent.py`).

```python
@fastapi_app.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(request: AgentRequest):
    volume.reload()
    try:
        result = process_agent_message(request.message)
        return {"response": result}
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

> **Explanation:**
>
> - The `/agent/chat` endpoint receives a JSON payload with the user’s message.
> - It reloads the volume (to get the latest database state) and calls `process_agent_message` to handle the query.
> - Errors are caught and returned as HTTP 500 responses.

#### 2. Google Token Endpoint

This endpoint accepts an OAuth access token from the frontend. It builds Google API credentials (testing that they are valid), and stores the token details.

```python
@fastapi_app.post("/auth/google/token")
def receive_token(token_data: TokenData):
    try:
        # Create credentials using the provided access token. this call will fail if we don't have the proper credentials
        creds = Credentials(
            token_data.access_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"]
        )
        # Store token details in SQLite.
        refresh_token = creds.refresh_token if creds.refresh_token else ""
        token_expiry = creds.expiry.isoformat() if creds.expiry else ""

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM google_tokens")
        cursor.execute(
            "INSERT INTO google_tokens (access_token, refresh_token, token_expiry) VALUES (?, ?, ?)",
            (token_data.access_token, refresh_token, token_expiry)
        )
        conn.commit()
        conn.close()
        volume.commit()

        return {
            "access_token": token_data.access_token,
            "refresh_token": refresh_token,
            "token_expiry": token_expiry
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

> **Explanation:**
>
> - We construct a `Credentials` object using the provided access token and environment variables for the client ID/secret.
> - The token information is saved in the `google_tokens` table for later use by our tool functions.
>   - Creds are valid for 1 hr with the way we have OAuth set up.

#### 3. Thread Management Endpoints

These endpoints allow you to delete the current agent thread (to force a new conversation) and retrieve the chat history:

```python
@fastapi_app.delete("/agent/thread")
def delete_agent_thread():
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
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/agent/history")
def get_agent_history():
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT thread_id FROM agent_threads ORDER BY updated_at DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {"messages": []}
        thread_id = row[0]

        messages = client.beta.threads.messages.list(
            thread_id=thread_id,
            order="asc"
        )
        chat_history = []
        if messages.data:
            for m in messages.data:
                role = m.role
                text = m.content[0].text.value if m.content and m.content[0].text.value else ""
                chat_history.append({"role": role, "text": text})
        return {"messages": chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

> **Explanation:**
>
> - Deleting a thread clears our persistent conversation, allowing the agent to start fresh.
> - The history endpoint uses the stored thread ID and OpenAI’s beta thread messaging API to list the conversation messages.

---

## 2. Integrating Google API Functions in `functions.py`

Our agent’s utility functions for scheduling meetings, sending emails, and reading calendar data are defined in `functions.py` ([full file for reference](https://github.com/nhein-tt/starter_template/blob/agent/backend_service/src/modal_app/functions.py)). These functions interface with Google’s Calendar and Gmail APIs.

### Getting Google Credentials

```python
def get_google_credentials() -> Credentials:
    """
    Retrieve the stored Google tokens from SQLite and return a Credentials object.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT access_token, refresh_token, token_expiry FROM google_tokens ORDER BY updated_at DESC LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        access_token, refresh_token, token_expiry = row
        return Credentials(
            access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv("GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET")
        )
    else:
        raise Exception("No stored Google credentials found.")
```

> **Explanation:**
>
> - This helper function loads the latest token from our `google_tokens` table and returns a `Credentials` object for authenticating Google API requests.

### Scheduling Meetings and Sending Emails

Here’s an example of our `schedule_meeting` function:

```python
def schedule_meeting(
    meeting_title: str,
    start_time: str,
    end_time: str,
    attendees: list = None,
    location: str = None,
):
    creds = get_google_credentials()
    service = build("calendar", "v3", credentials=creds)
    event = {
        "summary": meeting_title,
        "location": location or "TBD",
        "description": "Scheduled by your virtual EA",
        "start": {"dateTime": start_time, "timeZone": "UTC"},
        "end": {"dateTime": end_time, "timeZone": "UTC"},
        "attendees": [{"email": email} for email in attendees] if attendees else [],
        "reminders": {"useDefault": True},
    }
    created_event = service.events().insert(calendarId="primary", body=event).execute()
    return created_event
```

> **Explanation:**
>
> - Using the stored Google credentials, we build a Calendar API service.
> - The function creates an event with the provided parameters and returns the created event details.

Other functions like `send_email`, `read_emails`, `read_calendar`, and `edit_calendar` follow a similar pattern.  
Finally, the `run_function` helper maps a function name (from the agent’s tool call) to the appropriate Python function:

```python
def run_function(name: str, args: dict):
    if name == "schedule_meeting":
        return schedule_meeting(
            meeting_title=args["meeting_title"],
            start_time=args["start_time"],
            end_time=args["end_time"],
            attendees=args.get("attendees"),
            location=args.get("location"),
        )
    if name == "send_email":
        return send_email(
            recipient=args["recipient"],
            subject=args["subject"],
            body=args["body"]
        )
    if name == "read_emails":
        max_results = args.get("max_results", 5)
        return read_emails(max_results)
    if name == "read_calendar":
        max_results = args.get("max_results", 10)
        return read_calendar(max_results)
    if name == "edit_calendar":
        return edit_calendar(
            event_id=args["event_id"],
            updates=args["updates"]
        )
    return None
```

---

## 3. The AI Agent Logic in `agent.py`

This module is where our agent’s core logic lives. The assistant is set up with a prompt that instructs it to work as a virtual executive assistant, and it is provided with our tool definitions.

### Managing Conversation Threads

```python
def get_or_create_thread() -> str:
    """
    Retrieve an existing thread from the database or create a new one.
    Returns the thread ID.
    """
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id FROM agent_threads ORDER BY updated_at DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        thread_id = row[0]
    else:
        thread_obj = openai.beta.threads.create()
        thread_id = thread_obj.id
        cursor.execute("INSERT INTO agent_threads (thread_id) VALUES (?)", (thread_id,))
        conn.commit()
    conn.close()
    return thread_id
```

> **Explanation:**
>
> - The agent first checks if there’s an active thread stored in the database.
> - If not, it creates a new thread using OpenAI’s beta threads API and stores the new `thread_id` for persistence.

### Processing a User Message

The `process_agent_message` function is the heart of our agent. It:

1. Retrieves (or creates) a conversation thread.
2. Instantiates the assistant with our tool definitions (such as `schedule_meeting` and `send_email`).
3. Sends the user message to the thread.
4. Polls for a run, detects if any tool calls are required, executes them via our `run_function` helper, and finally retrieves the assistant’s final response.

```python
def process_agent_message(user_message: str) -> str:
    """
    Process a user message using the assistant.
    This function is fully stateless: it fetches (or creates) the conversation thread from the DB,
    sends the user message, polls the run, executes any tool calls in parallel, and returns the assistant's final response.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    thread_id = get_or_create_thread()
    assistant = client.beta.assistants.create(
        name="GoogleEA",
        instructions=CODE_PROMPT,
        tools=[
            {"type": "function", "function": functions[0]},  # schedule_meeting
            {"type": "function", "function": functions[1]},  # send_email
            {"type": "function", "function": functions[2]},  # read_emails
            {"type": "function", "function": functions[3]},  # read_calendar
            {"type": "function", "function": functions[4]},  # edit_calendar
        ],
        model="gpt-4o",
    )

    # Add the user's message to the thread.
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )

    # Initiate a run and poll for its completion.
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant.id,
    )

    if run.status == "requires_action":
        tool_outputs = []
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            name = tool.function.name
            args = json.loads(tool.function.arguments)
            result = run_function(name, args)
            tool_outputs.append({
                "tool_call_id": tool.id,
                "output": json.dumps(result)
            })

        if tool_outputs:
            run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
        else:
            return "No tool outputs generated."

    if run.status == "completed":
        messages = client.beta.threads.messages.list(
            thread_id=thread_id,
            order="asc"
        )
        if messages.data:
            last_message = messages.data[-1].content[0].text
            return last_message.value
        else:
            return "No messages found in thread."
    else:
        return f"Run status: {run.status}"
```

> **Explanation:**
>
> - The agent uses a prompt (`CODE_PROMPT`) to instruct the LLM to use the available tools.
> - It creates and polls a “run” of the conversation.
> - If tool calls are triggered, it executes each call (e.g., scheduling a meeting) and submits the outputs back to the assistant for further processing.
> - Finally, it retrieves and returns the assistant’s final message.

---

## 4. The Frontend: Executive Assistant Dashboard

In this section, we’ll examine the complete code for our Executive Assistant Dashboard—a React component that ties together all the frontend functionality of our AI-powered virtual assistant.
This dashboard not only handles the chat interface for interacting with the agent but also manages Google authentication, loads the required API scripts dynamically, and provides thread management for persistent conversations.

Before looking at all of the code below for the component, make sure you have all of your dependencies set up correctly. You will need to open up your `tailwind.config.js` file, and go to the `plugins` key, and make sure it looks like this:

```js
  plugins: [require("@tailwindcss/typography"), require("tailwindcss-animate")],
```

We use the typography plugin to render the markdown that we'll see output by the chat bot at times.

Then, install the `react-markdown` package, alongside the typography plugin, and all of the components we'll need by running:

```bash
bun add react-markdown
bun add -D @tailwindcss/typography
bunx --bun shadcn@latest add badge button calendar card input scroll-area separator table tabs toast
```

Below is the full code snippet for `EADashboard.tsx`:

```tsx
// src/components/EADashboard.tsx
import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Calendar, Mail, Send, Loader2, LogOut } from "lucide-react";
import Markdown from "react-markdown";
import { useToast } from "@/hooks/use-toast";

// Environment variables for Google authentication
const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID as string;
const API_KEY = import.meta.env.VITE_GOOGLE_API_KEY as string;
const SCOPES = import.meta.env.VITE_GOOGLE_SCOPES as string;

// Extend the global window object for Google API types
declare global {
  interface Window {
    gapi: any;
    google: any;
  }
}

// Define our ChatMessage and GoogleTokens types
interface ChatMessage {
  role: string;
  text: string;
}

interface GoogleTokens {
  access_token: string;
  refresh_token: string;
  token_expiry: string;
}

const ExecutiveAssistant = () => {
  // Chat state
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Google API state
  const [gapiLoaded, setGapiLoaded] = useState(false);
  const [gisLoaded, setGisLoaded] = useState(false);
  const [tokenClient, setTokenClient] = useState<any>(null);
  const [authorized, setAuthorized] = useState(false);
  const [googleTokens, setGoogleTokens] = useState<GoogleTokens | null>(null);
  const [activeTab, setActiveTab] = useState("chat");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  const modalUrl = import.meta.env.VITE_MODAL_URL as string;

  // Load Google API scripts on mount
  useEffect(() => {
    const gapiScript = document.createElement("script");
    gapiScript.src = "https://apis.google.com/js/api.js";
    gapiScript.async = true;
    gapiScript.defer = true;
    gapiScript.onload = () => window.gapi.load("client", initializeGapiClient);
    document.body.appendChild(gapiScript);

    const gisScript = document.createElement("script");
    gisScript.src = "https://accounts.google.com/gsi/client";
    gisScript.async = true;
    gisScript.defer = true;
    gisScript.onload = gisLoadedCallback;
    document.body.appendChild(gisScript);

    return () => {
      document.body.removeChild(gapiScript);
      document.body.removeChild(gisScript);
    };
  }, []);

  // Initialize the Google API client
  const initializeGapiClient = async () => {
    try {
      await window.gapi.client.init({
        apiKey: API_KEY,
      });
      setGapiLoaded(true);
      maybeEnableButtons();
    } catch (error) {
      console.error("Error initializing GAPI client:", error);
      toast({
        variant: "destructive",
        description: "Failed to initialize Google Calendar",
      });
    }
  };

  // Initialize Google Identity Services
  const gisLoadedCallback = () => {
    const client = window.google.accounts.oauth2.initTokenClient({
      client_id: CLIENT_ID,
      scope: SCOPES,
      callback: "", // Callback will be set in handleAuthClick
    });
    setTokenClient(client);
    setGisLoaded(true);
    maybeEnableButtons();
  };

  // Once both Google APIs are loaded, enable authentication buttons
  const maybeEnableButtons = () => {
    if (gapiLoaded && gisLoaded) {
      console.log("Google APIs initialized successfully");
    }
  };

  // Handle Google authentication (connect)
  const handleAuthClick = () => {
    if (!tokenClient) {
      toast({
        variant: "destructive",
        description: "Google authentication not ready",
      });
      return;
    }
    tokenClient.callback = async (resp: any) => {
      if (resp.error) {
        toast({
          variant: "destructive",
          description: "Google authentication failed",
        });
        return;
      }
      try {
        const tokenResponse = await fetch(`${modalUrl}/auth/google/token`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ access_token: resp.access_token }),
        });
        const tokens = await tokenResponse.json();
        setGoogleTokens(tokens);
        setAuthorized(true);
        toast({ description: "Successfully connected to Google Calendar" });
      } catch (err) {
        console.error("Error exchanging token:", err);
        toast({
          variant: "destructive",
          description: "Failed to connect to Google Calendar",
        });
      }
    };
    tokenClient.requestAccessToken({ prompt: "consent" });
  };

  // Handle Google sign-out (disconnect)
  const handleSignoutClick = () => {
    const token = window.gapi.client.getToken();
    if (token !== null) {
      window.google.accounts.oauth2.revoke(token.access_token);
      window.gapi.client.setToken("");
      setAuthorized(false);
      setGoogleTokens(null);
      toast({ description: "Disconnected from Google Calendar" });
    }
  };

  // Fetch chat history from our agent's backend
  const fetchChatHistory = async () => {
    try {
      const response = await fetch(`${modalUrl}/agent/history`);
      if (!response.ok) throw new Error("Failed to fetch chat history");
      const data = await response.json();
      setChatHistory(data.messages);
    } catch (err) {
      toast({
        variant: "destructive",
        description: "Failed to fetch chat history",
      });
    }
  };

  useEffect(() => {
    fetchChatHistory();
  }, []);

  // Automatically scroll to the bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // Handle chat form submission
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!message.trim()) return;
    setIsLoading(true);
    try {
      const userMessage = { role: "user", text: message };
      setChatHistory((prev) => [...prev, userMessage]);
      const response = await fetch(`${modalUrl}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      if (!response.ok) throw new Error("Failed to get agent response");
      const data = await response.json();
      const assistantMessage = { role: "assistant", text: data.response };
      setChatHistory((prev) => [...prev, assistantMessage]);
      setMessage("");
    } catch (err) {
      toast({
        variant: "destructive",
        description: "Failed to get agent response",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Reset the conversation thread
  const handleResetThread = async () => {
    try {
      const response = await fetch(`${modalUrl}/agent/thread`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error("Failed to reset thread");
      setChatHistory([]);
      toast({ description: "Chat thread reset successfully" });
    } catch (err) {
      toast({
        variant: "destructive",
        description: "Failed to reset chat thread",
      });
    }
  };

  // Component to render each chat message
  const MessageBubble = ({ message }: { message: ChatMessage }) => {
    const isUser = message.role === "user";
    return (
      <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
        <div
          className={`max-w-3/4 p-3 rounded-lg ${
            isUser
              ? "bg-blue-600 text-white rounded-br-none"
              : "bg-gray-100 text-gray-900 rounded-bl-none"
          }`}
        >
          <Markdown>{message.text}</Markdown>
        </div>
      </div>
    );
  };

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <Tabs defaultValue="chat" className="w-full" onValueChange={setActiveTab}>
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">Executive Assistant</h1>
          <div className="flex gap-2">
            {!authorized ? (
              <Button
                onClick={handleAuthClick}
                className="flex items-center gap-2"
                disabled={!gapiLoaded || !gisLoaded}
              >
                <Calendar className="w-4 h-4" /> Connect Google
              </Button>
            ) : (
              <Button
                onClick={handleSignoutClick}
                variant="outline"
                className="flex items-center gap-2"
              >
                <LogOut className="w-4 h-4" /> Disconnect
              </Button>
            )}
            <TabsList>
              <TabsTrigger value="chat" className="flex items-center gap-2">
                <Mail className="w-4 h-4" /> Chat
              </TabsTrigger>
            </TabsList>
          </div>
        </div>

        <TabsContent value="chat" className="mt-0">
          <Card>
            <CardContent className="p-6">
              <ScrollArea className="h-[600px] pr-4">
                {chatHistory.map((msg, idx) => (
                  <MessageBubble key={idx} message={msg} />
                ))}
                <div ref={messagesEndRef} />
              </ScrollArea>
              <form onSubmit={handleSubmit} className="flex gap-2 mt-4">
                <Input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Ask your assistant anything..."
                  className="flex-1"
                  disabled={isLoading}
                />
                <Button
                  type="submit"
                  disabled={isLoading || !message.trim()}
                  className="flex items-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" /> Thinking...
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" /> Send
                    </>
                  )}
                </Button>
              </form>
              <div className="flex justify-end mt-4">
                <Button
                  onClick={handleResetThread}
                  variant="outline"
                  className="text-sm"
                >
                  Reset Thread
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ExecutiveAssistant;
```

---

### How This Component Works

- **Google API Integration:**

  - The component dynamically loads both the Google API Client Library and the Google Identity Services library on mount.
  - Once loaded, it initializes the clients and enables the “Connect Google” button.

- **Authentication Handling:**

  - When the user clicks “Connect Google,” OAuth is triggered, and the resulting access token is sent to our backend via the `/auth/google/token` endpoint.
  - The UI updates to reflect a connected state, and a “Disconnect” button appears to allow sign-out.

- **Chat Interface:**

  - The chat area displays conversation history (fetched from `/agent/history`) using a scrollable view with Markdown-rendered message bubbles.
  - When the user sends a message (handled by `handleSubmit`), it is posted to the `/agent/chat` endpoint, and the response from the AI agent is appended to the chat history.

- **Thread Management:**
  - Users can reset the conversation thread by clicking “Reset Thread,” which clears the stored thread from our SQLite database so the next conversation starts fresh.

## Bringing It All Together

In this post, we extended our starter template to create an AI-powered executive assistant by:

- **Reusing the Base Architecture:**  
  We continued to use Modal functions, FastAPI endpoints, and SQLite persistence from our earlier posts.
- **Leveraging Advanced Features:**  
  The agent uses RAG-inspired code generation and function calling to dynamically decide when to execute tasks such as scheduling meetings or sending emails.
- **Integrating Third-Party APIs:**  
  With helper functions for Google Calendar and Gmail, our agent can interact with real-world services.

This modular, extensible design means you can easily add more tools or refine existing ones to suit your needs.

---

## Conclusion and Next Steps

You now have a fully functional AI agent that acts as a virtual executive assistant—capable of managing conversations, interfacing with Google APIs, and executing dynamic tool calls. Here are some ideas for what to try next:

- **Expand the Toolset:**  
  Add additional functions (e.g., fetching task lists or integrating with other services).
- **Improve Error Handling:**  
  Enhance the robustness of your endpoints and add retries or fallbacks for external API calls.
- **Customize the Assistant Prompt:**  
  Tailor the agent’s instructions to better suit your workflow or domain-specific needs.
- **UI Enhancements:**  
  Refine the dashboard with more detailed views, notifications, or additional tabs for other functionalities.

Happy building!
As always, feel free to reach out with questions or suggestions, and check out the `agent` branch in our repository for a working reference implementation.
