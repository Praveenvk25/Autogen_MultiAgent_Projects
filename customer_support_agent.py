"""
AutoGen Multi-Agent: Customer Support System
=============================================
Uses: pyautogen 0.10.0+ / autogen-agentchat 0.7.x (NEW API)
Real-world example:
  - SupportAgent : drafts a reply to the customer
  - QAAgent      : reviews and approves or improves it

HOW TO RUN:
  1. Make sure your venv is active
  2. Set your OpenAI API key below
  3. python customer_support_agent.py
"""

import asyncio
import threading
import tkinter as tk
from tkinter import scrolledtext

# NEW import style for pyautogen 0.10.0+
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ─────────────────────────────────────────
# 1. YOUR API KEY — paste it here
# ─────────────────────────────────────────
OPENAI_API_KEY = ""   # <-- replace with your real key

# ─────────────────────────────────────────
# 2. MODEL CLIENT
# ─────────────────────────────────────────
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
)

# ─────────────────────────────────────────
# 3. DEFINE AGENTS
# ─────────────────────────────────────────
support_agent = AssistantAgent(
    name="SupportAgent",
    model_client=model_client,
    system_message="""You are a friendly and professional customer support agent.
When given a customer query, draft a clear, empathetic, and helpful response.
Keep it under 5 sentences. Do NOT write APPROVED or TERMINATE.""",
)

qa_agent = AssistantAgent(
    name="QAAgent",
    model_client=model_client,
    system_message="""You are a senior QA reviewer for customer support responses.
Review the SupportAgent's reply. If it is good, respond with:
  APPROVED. TERMINATE
If it needs improvement, rewrite it and explain briefly why.""",
)

# ─────────────────────────────────────────
# 4. CUSTOMER QUERY
# ─────────────────────────────────────────
CUSTOMER_QUERY = (
    "Hi, I ordered a laptop 5 days ago (Order #45231) and haven't received "
    "a shipping confirmation yet. Can you help me find out what's happening?"
)

# ─────────────────────────────────────────
# 5. RUN THE TEAM
# ─────────────────────────────────────────
async def run_agents(on_message):
    termination = TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat(
        participants=[support_agent, qa_agent],
        termination_condition=termination,
        max_turns=6,
    )

    on_message("Customer", CUSTOMER_QUERY)

    async for event in team.run_stream(task=CUSTOMER_QUERY):
        # TaskResult fires at the very end — skip it
        if hasattr(event, "messages"):
            continue
        # Each agent message has .source and .content
        if hasattr(event, "source") and hasattr(event, "content"):
            sender = event.source
            content = event.content
            if isinstance(content, str) and content.strip():
                on_message(sender, content)

# ─────────────────────────────────────────
# 6. TKINTER POPUP WINDOW
# ─────────────────────────────────────────
def launch_window():
    window = tk.Tk()
    window.title("AutoGen Multi-Agent — Customer Support")
    window.geometry("860x600")
    window.configure(bg="#1e1e2e")

    tk.Label(
        window,
        text="AutoGen Multi-Agent Conversation",
        font=("Helvetica", 14, "bold"),
        fg="#cdd6f4", bg="#1e1e2e", pady=10,
    ).pack(fill="x")

    chat = scrolledtext.ScrolledText(
        window, wrap=tk.WORD, font=("Courier New", 11),
        bg="#181825", fg="#cdd6f4", insertbackground="white",
        borderwidth=0, padx=12, pady=8,
    )
    chat.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    chat.tag_config("Customer",      foreground="#89b4fa", font=("Courier New", 11, "bold"))
    chat.tag_config("SupportAgent",  foreground="#a6e3a1", font=("Courier New", 11, "bold"))
    chat.tag_config("QAAgent",       foreground="#fab387", font=("Courier New", 11, "bold"))
    chat.tag_config("System",        foreground="#6c7086", font=("Courier New", 10, "italic"))
    chat.tag_config("body",          foreground="#cdd6f4")

    def append(sender, content):
        def _w():
            chat.insert(tk.END, "─" * 62 + "\n", "System")
            tag = sender if sender in ("Customer", "SupportAgent", "QAAgent") else "System"
            chat.insert(tk.END, f"[{sender}]\n", tag)
            chat.insert(tk.END, content.strip() + "\n\n", "body")
            chat.see(tk.END)
        window.after(0, _w)

    def done():
        window.after(0, lambda: chat.insert(tk.END, "\n✅  Conversation complete.\n", "System"))

    def worker():
        append("System", "Starting agents... please wait.")
        try:
            asyncio.run(run_agents(append))
        except Exception as e:
            append("ERROR", str(e))
        done()

    threading.Thread(target=worker, daemon=True).start()
    window.mainloop()

# ─────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    launch_window()