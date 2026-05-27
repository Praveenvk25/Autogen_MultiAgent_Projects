"""
AI Job Application Pipeline — Multi-Agent System
=================================================
Uses: pyautogen 0.10.0+ / autogen-agentchat 0.7.x

4 Agents:
  1. JobAnalyserAgent  — reads job description, extracts keywords & requirements
  2. CVRewriterAgent   — rewrites your CV to match the job
  3. CoverLetterAgent  — writes a personalised cover letter
  4. MatchScoreAgent   — scores your CV vs job (0-100%) with tips

Supports CV upload: .txt / .pdf / .docx

HOW TO RUN:
  1. Activate venv
  2. pip install pyautogen openai autogen-ext[openai] pypdf python-docx
  3. Set your OPENAI_API_KEY below
  4. python job_application_pipeline.py
"""

import asyncio
import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

# ── CV file reading (pdf / docx / txt) ──────────────────────────
def read_cv_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        elif ext == ".docx":
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs).strip()
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
    except Exception as e:
        return f"[Error reading file: {e}]"

# ── AutoGen imports ──────────────────────────────────────────────
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ────────────────────────────────────────────────────────────────
# 1. YOUR API KEY
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = ""   # <-- paste your key here

# ────────────────────────────────────────────────────────────────
# 2. MODEL CLIENT
# ────────────────────────────────────────────────────────────────
def make_client():
    return OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
    )

# ────────────────────────────────────────────────────────────────
# 3. AGENT DEFINITIONS
# ────────────────────────────────────────────────────────────────
def make_agents():
    client = make_client()

    job_analyser = AssistantAgent(
        name="JobAnalyserAgent",
        model_client=client,
        system_message="""You are an expert job description analyst.
When given a job description and a CV, extract:
- Job title and company (if mentioned)
- Top 10 must-have skills and keywords
- Key responsibilities
- Tone of the company (formal/startup/technical)
- Any red flags or special requirements

Format your output clearly with headers.
End your message with: [JobAnalyserAgent DONE]""",
    )

    cv_rewriter = AssistantAgent(
        name="CVRewriterAgent",
        model_client=client,
        system_message="""You are an expert CV/resume writer and ATS optimization specialist.
Using the job analysis provided by JobAnalyserAgent and the original CV:
- Rewrite the CV to highlight relevant experience
- Naturally incorporate the top keywords from the job description
- Strengthen bullet points using strong action verbs
- Ensure ATS (Applicant Tracking System) compatibility
- Keep the same structure but optimize the content
- Add a tailored Professional Summary at the top

Output the full rewritten CV clearly.
End your message with: [CVRewriterAgent DONE]""",
    )

    cover_letter = AssistantAgent(
        name="CoverLetterAgent",
        model_client=client,
        system_message="""You are an expert cover letter writer.
Using the job analysis and the rewritten CV:
- Write a compelling, personalised cover letter (3-4 paragraphs)
- Opening: Hook the reader, mention the role
- Middle: Connect 2-3 specific experiences to the job requirements
- Closing: Strong call to action
- Match the tone of the company (formal/startup/creative)
- Keep it under 350 words

Output the full cover letter.
End your message with: [CoverLetterAgent DONE]""",
    )

    match_scorer = AssistantAgent(
        name="MatchScoreAgent",
        model_client=client,
        system_message="""You are an expert recruitment analyst and ATS scoring specialist.
Analyse the original CV against the job description and provide:

MATCH SCORE: X/100

Breakdown:
- Skills match: X/30
- Experience match: X/25
- Keywords match: X/20
- Education match: X/15
- Overall presentation: X/10

Strengths (3 bullet points of what works well)
Gaps (3 bullet points of what's missing)
Quick wins (3 specific things to improve immediately)

Be specific and honest.
End your message with: TERMINATE""",
    )

    return job_analyser, cv_rewriter, cover_letter, match_scorer


# ────────────────────────────────────────────────────────────────
# 4. PIPELINE RUNNER
# ────────────────────────────────────────────────────────────────
async def run_pipeline(cv_text: str, job_desc: str, on_message, on_done):
    job_analyser, cv_rewriter, cover_letter, match_scorer = make_agents()

    task = f"""
===== JOB DESCRIPTION =====
{job_desc}

===== CANDIDATE CV =====
{cv_text}

Instructions for all agents:
1. JobAnalyserAgent: Analyse the job description and CV above
2. CVRewriterAgent: Rewrite the CV based on JobAnalyserAgent's analysis
3. CoverLetterAgent: Write a cover letter based on the analysis and rewritten CV
4. MatchScoreAgent: Score the original CV against the job description and give recommendations
"""

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(
        participants=[job_analyser, cv_rewriter, cover_letter, match_scorer],
        termination_condition=termination,
        max_turns=8,
    )

    async for event in team.run_stream(task=task):
        if hasattr(event, "source") and hasattr(event, "content"):
            if isinstance(event.content, str) and event.content.strip():
                on_message(event.source, event.content.strip())

    on_done()


# ────────────────────────────────────────────────────────────────
# 5. TKINTER UI
# ────────────────────────────────────────────────────────────────
class JobPipelineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Job Application Pipeline")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1e1e2e")
        self.cv_path = tk.StringVar(value="No file selected")
        self.cv_text = ""
        self.is_running = False
        self._build_ui()

    def _build_ui(self):
        root = self.root

        # ── Top header ──────────────────────────────────────────
        hdr = tk.Frame(root, bg="#181825", pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  AI Job Application Pipeline",
                 font=("Helvetica", 15, "bold"),
                 fg="#cdd6f4", bg="#181825").pack(side="left", padx=12)
        tk.Label(hdr, text="4 AI agents working for you",
                 font=("Helvetica", 10),
                 fg="#6c7086", bg="#181825").pack(side="left")

        # ── Agent status bar ─────────────────────────────────────
        status_frame = tk.Frame(root, bg="#181825", pady=6)
        status_frame.pack(fill="x")
        self.agent_labels = {}
        agents = [
            ("JobAnalyserAgent",  "#89b4fa", "Analyse JD"),
            ("CVRewriterAgent",   "#a6e3a1", "Rewrite CV"),
            ("CoverLetterAgent",  "#fab387", "Cover Letter"),
            ("MatchScoreAgent",   "#f38ba8", "Score Match"),
        ]
        for name, color, label in agents:
            frame = tk.Frame(status_frame, bg="#181825")
            frame.pack(side="left", padx=14)
            dot = tk.Label(frame, text="●", font=("Helvetica", 14),
                           fg="#45475a", bg="#181825")
            dot.pack(side="left")
            tk.Label(frame, text=f" {label}",
                     font=("Helvetica", 10), fg="#6c7086",
                     bg="#181825").pack(side="left")
            self.agent_labels[name] = (dot, color)

        # ── Main paned layout: left inputs | right output ────────
        pane = tk.PanedWindow(root, orient="horizontal",
                              bg="#1e1e2e", sashwidth=4,
                              sashrelief="flat")
        pane.pack(fill="both", expand=True, padx=0, pady=0)

        # LEFT PANEL
        left = tk.Frame(pane, bg="#1e1e2e")
        pane.add(left, minsize=320)

        tk.Label(left, text="STEP 1 — Upload your CV",
                 font=("Helvetica", 11, "bold"),
                 fg="#6c7086", bg="#1e1e2e").pack(anchor="w", padx=16, pady=(14, 4))

        cv_row = tk.Frame(left, bg="#1e1e2e")
        cv_row.pack(fill="x", padx=16, pady=(0, 4))
        tk.Button(cv_row, text="Browse file",
                  font=("Helvetica", 11),
                  bg="#313244", fg="#cdd6f4",
                  activebackground="#45475a",
                  relief="flat", padx=12, pady=5,
                  cursor="hand2",
                  command=self._browse_cv).pack(side="left")
        tk.Label(cv_row, textvariable=self.cv_path,
                 font=("Helvetica", 10), fg="#6c7086",
                 bg="#1e1e2e", wraplength=180,
                 justify="left").pack(side="left", padx=8)

        self.cv_preview = scrolledtext.ScrolledText(
            left, height=8, font=("Courier New", 10),
            bg="#181825", fg="#6c7086",
            insertbackground="white", relief="flat",
            padx=8, pady=6, state="disabled",
            wrap=tk.WORD)
        self.cv_preview.pack(fill="x", padx=16, pady=(0, 12))

        tk.Label(left, text="STEP 2 — Paste job description",
                 font=("Helvetica", 11, "bold"),
                 fg="#6c7086", bg="#1e1e2e").pack(anchor="w", padx=16, pady=(0, 4))

        self.jd_box = scrolledtext.ScrolledText(
            left, height=14, font=("Courier New", 10),
            bg="#181825", fg="#cdd6f4",
            insertbackground="#cdd6f4", relief="flat",
            padx=8, pady=6, wrap=tk.WORD)
        self.jd_box.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.jd_box.insert("1.0",
            "Paste the full job description here...\n\n"
            "Example:\nWe are looking for a Senior Python Developer...\n"
            "Requirements: 5+ years Python, AWS, REST APIs, SQL...")

        self.run_btn = tk.Button(
            left,
            text="Run all 4 agents  →",
            font=("Helvetica", 12, "bold"),
            bg="#7f77dd", fg="white",
            activebackground="#534AB7",
            relief="flat", pady=10,
            cursor="hand2",
            command=self._run_pipeline)
        self.run_btn.pack(fill="x", padx=16, pady=(0, 16))

        # RIGHT PANEL — tabbed output
        right = tk.Frame(pane, bg="#1e1e2e")
        pane.add(right, minsize=500)

        tk.Label(right, text="STEP 3 — Agent outputs",
                 font=("Helvetica", 11, "bold"),
                 fg="#6c7086", bg="#1e1e2e").pack(anchor="w", padx=16, pady=(14, 6))

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                        background="#1e1e2e",
                        borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background="#313244",
                        foreground="#6c7086",
                        padding=[12, 6],
                        font=("Helvetica", 10))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", "#45475a")],
                  foreground=[("selected", "#cdd6f4")])

        nb = ttk.Notebook(right, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        tab_configs = [
            ("Live log",       "#cdd6f4"),
            ("Job analysis",   "#89b4fa"),
            ("Rewritten CV",   "#a6e3a1"),
            ("Cover letter",   "#fab387"),
            ("Match score",    "#f38ba8"),
        ]
        self.tabs = {}
        for title, color in tab_configs:
            frame = tk.Frame(nb, bg="#181825")
            nb.add(frame, text=title)
            box = scrolledtext.ScrolledText(
                frame, wrap=tk.WORD,
                font=("Courier New", 11),
                bg="#181825", fg=color,
                insertbackground="white",
                borderwidth=0, padx=12, pady=8,
                state="disabled")
            box.pack(fill="both", expand=True)
            self.tabs[title] = box

        self.nb = nb

    # ── file browser ────────────────────────────────────────────
    def _browse_cv(self):
        path = filedialog.askopenfilename(
            title="Select your CV",
            filetypes=[
                ("All supported", "*.pdf *.docx *.txt"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("Text files", "*.txt"),
            ]
        )
        if path:
            self.cv_path.set(os.path.basename(path))
            self.cv_text = read_cv_file(path)
            self._preview_cv()

    def _preview_cv(self):
        box = self.cv_preview
        box.config(state="normal")
        box.delete("1.0", tk.END)
        preview = self.cv_text[:600] + ("..." if len(self.cv_text) > 600 else "")
        box.insert("1.0", preview)
        box.config(state="disabled")

    # ── write to a tab ──────────────────────────────────────────
    def _write(self, tab_name: str, text: str, color: str = None):
        def _do():
            box = self.tabs[tab_name]
            box.config(state="normal")
            if color:
                tag = f"col_{color.replace('#','')}"
                box.tag_config(tag, foreground=color)
                box.insert(tk.END, text, tag)
            else:
                box.insert(tk.END, text)
            box.see(tk.END)
            box.config(state="disabled")
        self.root.after(0, _do)

    def _clear_tab(self, tab_name: str):
        def _do():
            box = self.tabs[tab_name]
            box.config(state="normal")
            box.delete("1.0", tk.END)
            box.config(state="disabled")
        self.root.after(0, _do)

    # ── activate agent dot ───────────────────────────────────────
    def _set_agent_active(self, name: str, active: bool):
        def _do():
            if name in self.agent_labels:
                dot, color = self.agent_labels[name]
                dot.config(fg=color if active else "#45475a")
        self.root.after(0, _do)

    def _deactivate_all_agents(self):
        for name in self.agent_labels:
            self._set_agent_active(name, False)

    # ── map agent name to tab ────────────────────────────────────
    TAB_MAP = {
        "JobAnalyserAgent":  "Job analysis",
        "CVRewriterAgent":   "Rewritten CV",
        "CoverLetterAgent":  "Cover letter",
        "MatchScoreAgent":   "Match score",
    }
    AGENT_COLORS = {
        "JobAnalyserAgent":  "#89b4fa",
        "CVRewriterAgent":   "#a6e3a1",
        "CoverLetterAgent":  "#fab387",
        "MatchScoreAgent":   "#f38ba8",
    }

    # ── on_message callback ──────────────────────────────────────
    def _on_message(self, sender: str, content: str):
        color = self.AGENT_COLORS.get(sender, "#cdd6f4")

        # Deactivate previous, activate current
        self._deactivate_all_agents()
        self._set_agent_active(sender, True)

        # Write to live log
        self._write("Live log",
                    f"\n{'─'*55}\n[{sender}]\n", color)
        self._write("Live log", content + "\n")

        # Write to dedicated tab
        tab = self.TAB_MAP.get(sender)
        if tab:
            self._clear_tab(tab)
            self._write(tab, content)

    # ── on_done callback ─────────────────────────────────────────
    def _on_done(self):
        self._deactivate_all_agents()
        self._write("Live log",
                    "\n✅  All 4 agents have completed. "
                    "Check each tab for detailed outputs.\n",
                    "#a6e3a1")
        def _re_enable():
            self.run_btn.config(
                state="normal",
                text="Run all 4 agents  →",
                bg="#7f77dd")
            self.is_running = False
        self.root.after(0, _re_enable)

    # ── run pipeline ─────────────────────────────────────────────
    def _run_pipeline(self):
        if self.is_running:
            return

        # Validate inputs
        cv = self.cv_text.strip()
        jd = self.jd_box.get("1.0", tk.END).strip()

        if not cv:
            self._show_error("Please upload your CV first (Browse file button).")
            return
        if not jd or jd.startswith("Paste the full job description"):
            self._show_error("Please paste a job description in the left panel.")
            return
        if "sk-..." in OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
            self._show_error("Please set your OPENAI_API_KEY in the Python file (line 58).")
            return

        self.is_running = True
        self.run_btn.config(state="disabled",
                            text="Agents running...",
                            bg="#45475a")

        # Clear all output tabs
        for tab_name in self.tabs:
            self._clear_tab(tab_name)

        self._write("Live log",
                    "Starting AI Job Application Pipeline...\n"
                    "4 agents will run in sequence.\n"
                    "This takes 60-90 seconds.\n\n",
                    "#6c7086")

        def worker():
            asyncio.run(run_pipeline(
                cv, jd,
                self._on_message,
                self._on_done,
            ))

        threading.Thread(target=worker, daemon=True).start()

    def _show_error(self, msg: str):
        self._write("Live log", f"\n⚠  {msg}\n", "#f38ba8")
        self.nb.select(0)


# ────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = JobPipelineApp(root)
    root.mainloop()