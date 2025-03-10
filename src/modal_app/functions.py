# src/modal_app/functions.py
import os
import sqlite3
import datetime
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from .common import DB_PATH


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
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        )
    else:
        raise Exception("No stored Google credentials found.")


def schedule_meeting(
    meeting_title: str,
    start_time: str,
    end_time: str,
    attendees: list = None,
    location: str = None,
):
    """
    Schedule a meeting in Google Calendar using stored credentials.
    start_time and end_time must be ISO 8601 strings.
    """
    creds = get_google_credentials()
    service = build("calendar", "v3", credentials=creds)
    event = {
        "summary": meeting_title,
        "location": location or "TBD",
        "description": "Scheduled by your virtual EA",
        "start": {
            "dateTime": start_time,
            "timeZone": "UTC",
        },
        "end": {
            "dateTime": end_time,
            "timeZone": "UTC",
        },
        "attendees": [{"email": email} for email in attendees] if attendees else [],
        "reminders": {"useDefault": True},
    }
    created_event = service.events().insert(calendarId="primary", body=event).execute()
    return created_event


def send_email(recipient: str, subject: str, body: str):
    """
    Send an email using the Gmail API.
    Builds a MIME message, encodes it in base64, and sends it.
    """
    creds = get_google_credentials()
    service = build("gmail", "v1", credentials=creds)
    message = MIMEText(body)
    message["to"] = recipient
    message["subject"] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent_message = (
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw_message})
        .execute()
    )
    return sent_message


def read_emails(max_results: int = 5):
    """
    Read unread emails from the user's Gmail inbox.
    Returns a list of email summaries including subject and snippet.
    """
    creds = get_google_credentials()
    service = build("gmail", "v1", credentials=creds)
    response = (
        service.users()
        .messages()
        .list(userId="me", labelIds=["INBOX", "UNREAD"], maxResults=max_results)
        .execute()
    )
    messages = response.get("messages", [])
    email_list = []
    for msg in messages:
        msg_detail = (
            service.users()
            .messages()
            .get(userId="me", id=msg["id"], format="full")
            .execute()
        )
        headers = msg_detail.get("payload", {}).get("headers", [])
        subject = ""
        for header in headers:
            if header["name"] == "Subject":
                subject = header["value"]
                break
        snippet = msg_detail.get("snippet", "")
        email_list.append({"subject": subject, "snippet": snippet})
    return email_list


def read_calendar(max_results: int = 10):
    """
    Read upcoming events from the user's Google Calendar.
    Returns a list of event summaries and their start times.
    """
    creds = get_google_credentials()
    service = build("calendar", "v3", credentials=creds)
    now = datetime.datetime.utcnow().isoformat() + "Z"
    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])
    event_list = []
    for event in events:
        start = event.get("start", {}).get(
            "dateTime", event.get("start", {}).get("date")
        )
        event_list.append({"summary": event.get("summary", "No Title"), "start": start})
    return event_list


def edit_calendar(event_id: str, updates: dict):
    """
    Edit an existing calendar event.
    'updates' may include new summary, start_time, end_time, and location.
    Returns the updated event.
    """
    creds = get_google_credentials()
    service = build("calendar", "v3", credentials=creds)
    event = {}
    if "summary" in updates:
        event["summary"] = updates["summary"]
    if "start_time" in updates:
        event["start"] = {"dateTime": updates["start_time"], "timeZone": "UTC"}
    if "end_time" in updates:
        event["end"] = {"dateTime": updates["end_time"], "timeZone": "UTC"}
    if "location" in updates:
        event["location"] = updates["location"]
    updated_event = (
        service.events()
        .patch(calendarId="primary", eventId=event_id, body=event)
        .execute()
    )
    return updated_event


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
            recipient=args["recipient"], subject=args["subject"], body=args["body"]
        )
    if name == "read_emails":
        max_results = args.get("max_results", 5)
        return read_emails(max_results)
    if name == "read_calendar":
        max_results = args.get("max_results", 10)
        return read_calendar(max_results)
    if name == "edit_calendar":
        return edit_calendar(event_id=args["event_id"], updates=args["updates"])
    return None


# Define the function metadata (tools/skills) for the agent.
functions = [
    {
        "name": "schedule_meeting",
        "description": "Schedule a meeting in Google Calendar.",
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_title": {
                    "type": "string",
                    "description": "Title of the meeting",
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format",
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in ISO 8601 format",
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of attendee email addresses",
                },
                "location": {
                    "type": "string",
                    "description": "Optional meeting location",
                },
            },
            "required": ["meeting_title", "start_time", "end_time"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email via the Gmail API.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Recipient email address",
                },
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body content"},
            },
            "required": ["recipient", "subject", "body"],
        },
    },
    {
        "name": "read_emails",
        "description": "Read unread emails from the Gmail inbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "number",
                    "description": "Maximum number of emails to return (default is 5)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_calendar",
        "description": "Read upcoming events from the Google Calendar.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "number",
                    "description": "Maximum number of events to return (default is 10)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "edit_calendar",
        "description": "Edit an existing calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The ID of the event to update",
                },
                "updates": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "New title for the event",
                        },
                        "start_time": {
                            "type": "string",
                            "description": "New start time in ISO 8601 format",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "New end time in ISO 8601 format",
                        },
                        "location": {
                            "type": "string",
                            "description": "New location for the event",
                        },
                    },
                    "required": [],
                },
            },
            "required": ["event_id", "updates"],
        },
    },
]
