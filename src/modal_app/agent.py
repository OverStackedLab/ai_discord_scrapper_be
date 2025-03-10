# src/modal_app/agent.py
import time
import json
import os
import sqlite3
from openai import OpenAI
from .common import DB_PATH
from .functions import (
    run_function,
    functions,
)  # This module defines schedule_meeting & send_email, etc.

# Set your OpenAI API key from your environment.

# Define the agent prompt that instructs the assistant to work as a virtual executive assistant.
CODE_PROMPT = (
    "You are a virtual executive assistant that helps schedule meetings and send emails using Google APIs. "
    "When a user instructs you to schedule a meeting, you must call the 'schedule_meeting' function with appropriate parameters. "
    "Similarly, if the user asks you to send an email, call the 'send_email' function. "
    "Use clear, concise language. When a function call is needed, output a JSON-formatted call."
)


def get_or_create_thread() -> str:
    """
    Retrieve an existing thread from the database or create a new one.
    Returns the thread ID.
    """
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT thread_id FROM agent_threads ORDER BY updated_at DESC LIMIT 1"
    )
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


def process_agent_message(user_message: str) -> str:
    """
    Process a user message using the assistant.
    This function is fully stateless: it fetches (or creates) the conversation thread from the DB,
    sends the user message, polls the run, executes any tool calls in parallel (using the new SDK helpers),
    and returns the assistant's final response.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # Retrieve (or create) the persistent thread ID.
    thread_id = get_or_create_thread()
    # Create the assistant with the tool definitions.
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
        thread_id=thread_id, role="user", content=user_message
    )

    # Initiate a run and poll for its completion.
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant.id,
    )

    # If the run requires action (i.e. tool calls), process them.
    if run.status == "requires_action":
        tool_outputs = []
        # Iterate over each tool call triggered in parallel.
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            name = tool.function.name
            # Parse the JSON-formatted arguments.
            args = json.loads(tool.function.arguments)
            # Execute the corresponding tool function (e.g., schedule_meeting or send_email).
            result = run_function(name, args)
            # Append the result in the format expected by the SDK.
            tool_outputs.append({"tool_call_id": tool.id, "output": json.dumps(result)})

        if tool_outputs:
            # Submit all tool outputs at once and poll for the run’s completion.
            run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
            )
        else:
            return "No tool outputs generated."

    # After the run completes, retrieve all messages from the thread.
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        if messages.data:
            # Return the final (latest) assistant message.
            last_message = messages.data[-1].content[0].text
            return last_message.value
        else:
            return "No messages found in thread."
    else:
        return f"Run status: {run.status}"
