"""
Human-in-the-Loop Chatbot: AI Travel Planner
=============================================
Uses: pyautogen 0.10.0+ / autogen-agentchat 0.7.x

Real-world example:
  YOU type your travel preferences
  TravelAgent asks follow-up questions and builds your itinerary
  The conversation is LIVE - you type, it replies in real time

HOW TO RUN:
  1. Activate your venv
  2. Set your OpenAI API key below
  3. python travel_planner_chatbot.py
  4. Type in the input box and press Send (or Enter)
"""

import asyncio
import threading
import tkinter as tk
from tkinter import scrolledtext

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ─────────────────────────────────────────
# 1. YOUR API KEY
# ─────────────────────────────────────────
OPENAI_API_KEY = ""   # <-- paste your key here

# ─────────────────────────────────────────
# 2. MODEL CLIENT
# ─────────────────────────────────────────
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
)

# ─────────────────────────────────────────
# 3. TRAVEL AGENT DEFINITION
# ─────────────────────────────────────────
travel_agent = AssistantAgent(
    name="TravelAgent",
    model_client=model_client,
    system_message="""You are an expert AI travel planner. Your job is to have a 
friendly conversation with the user to understand their travel preferences and 
then build them a detailed day-by-day itinerary.

Follow this conversation flow:
1. Greet the user warmly and ask where they want to go
2. Ask how many days they are travelling
3. Ask their budget level (budget / mid-range / luxury)
4. Ask what type of activities they enjoy (culture, adventure, food, relaxation, shopping)
5. Ask if they have any dietary restrictions or special needs
6. Once you have all the information, generate a detailed day-by-day itinerary

Rules:
- Ask ONE question at a time — never ask multiple questions together
- Be friendly, warm and conversational
- When generating the itinerary, be very specific with places, timings, and tips
- After delivering the itinerary, ask if they want to adjust anything
- Only say TERMINATE when the user says goodbye, thanks, or they are done
""",
)

# ─────────────────────────────────────────
# 4. CONVERSATION STATE
# ─────────────────────────────────────────
conversation_history = []
response_queue = asyncio.Queue()

# ─────────────────────────────────────────
# 5. ASYNC CHAT ENGINE
# ─────────────────────────────────────────
async def chat_turn(user_message, on_reply):
    """Send one user message and get the agent reply."""
    conversation_history.append({"role": "user", "content": user_message})

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(
        participants=[travel_agent],
        termination_condition=termination,
        max_turns=1,
    )

    full_reply = ""
    async for event in team.run_stream(task=user_message):
        if hasattr(event, "source") and hasattr(event, "content"):
            if event.source == "TravelAgent":
                content = event.content
                if isinstance(content, str) and content.strip():
                    full_reply = content.strip()

    if full_reply:
        conversation_history.append({"role": "assistant", "content": full_reply})
        on_reply(full_reply)

def run_async_chat(user_message, on_reply, on_done):
    """Bridge between tkinter (sync) and asyncio (async)."""
    async def _run():
        await chat_turn(user_message, on_reply)
        on_done()
    asyncio.run(_run())

# ─────────────────────────────────────────
# 6. TKINTER CHAT WINDOW
# ─────────────────────────────────────────
def launch_chatbot():
    window = tk.Tk()
    window.title("AI Travel Planner — Chat with your agent")
    window.geometry("780x620")
    window.configure(bg="#1e1e2e")

    # ── Header ──
    header_frame = tk.Frame(window, bg="#181825", pady=10)
    header_frame.pack(fill="x")
    tk.Label(
        header_frame,
        text="  AI Travel Planner",
        font=("Helvetica", 14, "bold"),
        fg="#cdd6f4", bg="#181825",
    ).pack(side="left", padx=12)
    tk.Label(
        header_frame,
        text="Human-in-the-loop chatbot",
        font=("Helvetica", 10),
        fg="#6c7086", bg="#181825",
    ).pack(side="left")

    # ── Chat display ──
    chat_display = scrolledtext.ScrolledText(
        window, wrap=tk.WORD, font=("Helvetica", 12),
        bg="#181825", fg="#cdd6f4", insertbackground="white",
        borderwidth=0, padx=14, pady=10, state="disabled",
    )
    chat_display.pack(fill="both", expand=True, padx=0, pady=0)

    # Color tags
    chat_display.tag_config("you_label",   foreground="#89b4fa", font=("Helvetica", 11, "bold"))
    chat_display.tag_config("you_text",    foreground="#b0c4de", font=("Helvetica", 12))
    chat_display.tag_config("agent_label", foreground="#a6e3a1", font=("Helvetica", 11, "bold"))
    chat_display.tag_config("agent_text",  foreground="#cdd6f4", font=("Helvetica", 12))
    chat_display.tag_config("system",      foreground="#6c7086", font=("Helvetica", 10, "italic"))
    chat_display.tag_config("typing",      foreground="#fab387", font=("Helvetica", 11, "italic"))

    def append_msg(label, label_tag, text, text_tag):
        chat_display.config(state="normal")
        chat_display.insert(tk.END, f"\n{label}\n", label_tag)
        chat_display.insert(tk.END, f"{text}\n", text_tag)
        chat_display.see(tk.END)
        chat_display.config(state="disabled")

    def append_system(text):
        chat_display.config(state="normal")
        chat_display.insert(tk.END, f"\n{text}\n", "system")
        chat_display.see(tk.END)
        chat_display.config(state="disabled")

    typing_label_id = [None]

    def show_typing():
        chat_display.config(state="normal")
        chat_display.insert(tk.END, "\nTravelAgent is typing...\n", "typing")
        typing_label_id[0] = chat_display.index("end-2l")
        chat_display.see(tk.END)
        chat_display.config(state="disabled")

    def hide_typing():
        chat_display.config(state="normal")
        try:
            chat_display.delete("end-3l", "end-1l")
        except Exception:
            pass
        chat_display.config(state="disabled")

    # ── Input area ──
    input_frame = tk.Frame(window, bg="#313244", pady=10, padx=12)
    input_frame.pack(fill="x", side="bottom")

    user_input = tk.Entry(
        input_frame,
        font=("Helvetica", 12),
        bg="#45475a", fg="#cdd6f4",
        insertbackground="#cdd6f4",
        relief="flat", bd=6,
    )
    user_input.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 10))

    send_btn = tk.Button(
        input_frame,
        text="Send",
        font=("Helvetica", 11, "bold"),
        bg="#7f77dd", fg="white",
        activebackground="#534AB7",
        relief="flat", padx=16, pady=6,
        cursor="hand2",
    )
    send_btn.pack(side="right")

    is_waiting = [False]

    def on_send(event=None):
        if is_waiting[0]:
            return
        msg = user_input.get().strip()
        if not msg:
            return

        user_input.delete(0, tk.END)
        append_msg("You", "you_label", msg, "you_text")
        send_btn.config(state="disabled", text="...")
        is_waiting[0] = True

        window.after(100, show_typing)

        def on_reply(reply):
            window.after(0, lambda: (
                hide_typing(),
                append_msg("TravelAgent", "agent_label", reply, "agent_text"),
            ))

        def on_done():
            window.after(0, lambda: (
                send_btn.config(state="normal", text="Send"),
                setattr(is_waiting, 0, False),
                user_input.focus(),
            ))
            is_waiting[0] = False

        threading.Thread(
            target=run_async_chat,
            args=(msg, on_reply, on_done),
            daemon=True,
        ).start()

    send_btn.config(command=on_send)
    user_input.bind("<Return>", on_send)

    # ── Welcome message ──
    append_system("Connected to TravelAgent. Start by telling the agent where you want to go!")
    append_msg(
        "TravelAgent", "agent_label",
        "Hello! Welcome to your personal AI Travel Planner! I'm here to help you plan the perfect trip. "
        "To get started — where in the world would you like to travel?",
        "agent_text",
    )

    user_input.focus()
    window.mainloop()

# ─────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    launch_chatbot()