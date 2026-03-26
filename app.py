"""
app.py
------
Mental Disorder Diagnosis – Python Desktop App (Tkinter)
University of Eastern Africa, Baraton

Features:
  - Clean, professional GUI
  - DASS-21 questionnaire (21 questions + demographics)
  - Real-time prediction using saved Random Forest model
  - Results screen with disorder probability and severity level
  - Option to retake assessment or exit
  - Graceful fallback if model not trained yet
"""

import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import os

# ── Try loading model; warn user if not yet trained ──────────────────────────
try:
    import joblib
    MODEL_LOADED = True
    _base = os.path.dirname(os.path.abspath(__file__))
    best_model     = joblib.load(os.path.join(_base, "models/best_model.pkl"))
    scaler         = joblib.load(os.path.join(_base, "models/scaler.pkl"))
    feature_names  = joblib.load(os.path.join(_base, "models/feature_names.pkl"))
    label_encoder  = joblib.load(os.path.join(_base, "models/label_encoder.pkl"))
except Exception as e:
    MODEL_LOADED = False
    _model_error = str(e)

# ── Constants ────────────────────────────────────────────────────────────────
APP_TITLE  = "MindCheck — Mental Disorder Screening Tool"
BG_DARK    = "#1A1A2E"
BG_CARD    = "#16213E"
BG_ACCENT  = "#0F3460"
CLR_TEAL   = "#00B4D8"
CLR_GREEN  = "#4CAF50"
CLR_ORANGE = "#FF9800"
CLR_RED    = "#F44336"
CLR_WHITE  = "#F0F0F0"
CLR_GRAY   = "#9E9E9E"

DASS21_QUESTIONS = [
    # (question_text, subscale)
    ("I found it hard to wind down",                                    "stress"),
    ("I was aware of dryness of my mouth",                              "anxiety"),
    ("I couldn't seem to experience any positive feeling at all",       "depression"),
    ("I experienced breathing difficulty",                              "anxiety"),
    ("I found it difficult to work up the initiative to do things",     "depression"),
    ("I tended to over-react to situations",                            "stress"),
    ("I experienced trembling (e.g. in the hands)",                     "anxiety"),
    ("I felt that I was using a lot of nervous energy",                 "stress"),
    ("I was worried about situations in which I might panic and look foolish", "anxiety"),
    ("I felt that I had nothing to look forward to",                    "depression"),
    ("I found myself getting agitated",                                 "stress"),
    ("I found it difficult to relax",                                   "stress"),
    ("I felt down-hearted and blue",                                    "depression"),
    ("I was intolerant of anything that kept me from getting on with what I was doing", "stress"),
    ("I felt I was close to panic",                                     "anxiety"),
    ("I was unable to become enthusiastic about anything",              "depression"),
    ("I felt I wasn't worth much as a person",                          "depression"),
    ("I felt that I was rather touchy",                                 "stress"),
    ("I was aware of the action of my heart in the absence of physical exertion", "anxiety"),
    ("I felt scared without any good reason",                           "anxiety"),
    ("I felt that life was meaningless",                                "depression"),
]

SCALE_LABELS = ["0 – Did not apply to me at all",
                "1 – Applied to me to some degree",
                "2 – Applied to me to a considerable degree",
                "3 – Applied to me very much"]

DEPRESSION_Q = [2,4,9,12,15,16,20]   # 0-indexed
ANXIETY_Q    = [1,3,6,8,14,18,19]
STRESS_Q     = [0,5,7,10,11,13,17]


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Class
# ─────────────────────────────────────────────────────────────────────────────
class MindCheckApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("860x680")
        self.resizable(True, True)
        self.configure(bg=BG_DARK)
        self.minsize(800, 600)

        # Custom fonts
        self.title_font   = font.Font(family="Helvetica", size=20, weight="bold")
        self.heading_font = font.Font(family="Helvetica", size=13, weight="bold")
        self.body_font    = font.Font(family="Helvetica", size=11)
        self.small_font   = font.Font(family="Helvetica", size=9)

        # State
        self.q_vars   = [tk.IntVar(value=-1) for _ in range(21)]
        self.age_var  = tk.StringVar()
        self.gen_var  = tk.StringVar(value="Male")
        self.slp_var  = tk.StringVar(value="7")
        self.exc_var  = tk.StringVar(value="3")

        self._show_home()

    # ── Utility ──────────────────────────────────────────────────────────────
    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _frame(self, parent, **kw):
        return tk.Frame(parent, bg=kw.get('bg', BG_DARK), **{k:v for k,v in kw.items() if k != 'bg'})

    def _label(self, parent, text, **kw):
        return tk.Label(parent, text=text, bg=kw.get('bg', BG_DARK),
                        fg=kw.get('fg', CLR_WHITE),
                        font=kw.get('font', self.body_font),
                        **{k:v for k,v in kw.items() if k not in ('bg','fg','font')})

    def _btn(self, parent, text, cmd, color=CLR_TEAL, fg=BG_DARK, **kw):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg=fg, font=self.heading_font,
                      relief='flat', cursor='hand2',
                      padx=20, pady=10, **kw)
        b.bind("<Enter>", lambda e: b.config(bg=self._lighten(color)))
        b.bind("<Leave>", lambda e: b.config(bg=color))
        return b

    @staticmethod
    def _lighten(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = [min(255, int(hex_color[i:i+2], 16) + 30) for i in (0, 2, 4)]
        return f"#{r:02x}{g:02x}{b:02x}"

    # ── Screen 1: Home ────────────────────────────────────────────────────────
    def _show_home(self):
        self._clear()

        # Header
        header = self._frame(self, bg=BG_ACCENT)
        header.pack(fill='x')
        self._label(header, "🧠  MindCheck", bg=BG_ACCENT, fg=CLR_TEAL,
                    font=self.title_font).pack(pady=(20, 4))
        self._label(header, "Mental Disorder Screening Tool — DASS-21",
                    bg=BG_ACCENT, fg=CLR_WHITE,
                    font=self.heading_font).pack(pady=(0, 4))
        self._label(header,
                    "University of Eastern Africa, Baraton  |  Aduma · Betto · Bolo",
                    bg=BG_ACCENT, fg=CLR_GRAY, font=self.small_font).pack(pady=(0, 16))

        # Body
        body = self._frame(self)
        body.pack(expand=True, fill='both', padx=60, pady=30)

        card = self._frame(body, bg=BG_CARD)
        card.pack(fill='both', expand=True, padx=0, pady=0)

        self._label(card, "Welcome", bg=BG_CARD, fg=CLR_TEAL,
                    font=self.heading_font).pack(pady=(30, 8))

        info = ("This tool uses a machine-learning model trained on DASS-21 responses\n"
                "to screen for symptoms of Depression, Anxiety and Stress.\n\n"
                "The assessment takes approximately 3–5 minutes.\n"
                "Please answer all 21 questions as honestly as possible.\n\n"
                "⚠  This is a screening aid — NOT a clinical diagnosis.\n"
                "    Always consult a qualified mental health professional.")
        self._label(card, info, bg=BG_CARD, fg=CLR_WHITE,
                    font=self.body_font, justify='center',
                    wraplength=620).pack(pady=10)

        if not MODEL_LOADED:
            self._label(card, f"⚠  Model not found. Run train_model.py first.\n({_model_error})",
                        bg=BG_CARD, fg=CLR_ORANGE, font=self.small_font,
                        wraplength=620).pack(pady=6)

        self._btn(card, "  Start Assessment  →", self._show_demographics,
                  color=CLR_TEAL).pack(pady=30)

    # ── Screen 2: Demographics ────────────────────────────────────────────────
    def _show_demographics(self):
        self._clear()
        self._header("Step 1 of 3 — Your Information")

        body = self._frame(self)
        body.pack(expand=True, fill='both', padx=60, pady=20)

        card = self._frame(body, bg=BG_CARD)
        card.pack(fill='both', expand=True)

        self._label(card, "Please provide some background information.",
                    bg=BG_CARD, fg=CLR_GRAY, font=self.body_font).pack(pady=(20,10))

        form = self._frame(card, bg=BG_CARD)
        form.pack(padx=80, pady=10)

        fields = [
            ("Age", self.age_var, "entry"),
            ("Gender", self.gen_var, ("Male", "Female", "Other")),
            ("Average sleep (hours/night)", self.slp_var, "entry"),
            ("Exercise days per week (0–7)", self.exc_var, "entry"),
        ]

        for label, var, widget_type in fields:
            row = self._frame(form, bg=BG_CARD)
            row.pack(fill='x', pady=6)
            self._label(row, label + ":", bg=BG_CARD, fg=CLR_WHITE,
                        font=self.body_font, width=35, anchor='w').pack(side='left')
            if widget_type == "entry":
                e = tk.Entry(row, textvariable=var, width=12,
                             bg=BG_ACCENT, fg=CLR_WHITE, insertbackground=CLR_WHITE,
                             font=self.body_font, relief='flat')
                e.pack(side='left', padx=(6,0))
            else:
                cb = ttk.Combobox(row, textvariable=var, values=widget_type,
                                  state='readonly', width=12, font=self.body_font)
                cb.pack(side='left', padx=(6,0))

        btn_row = self._frame(card, bg=BG_CARD)
        btn_row.pack(pady=30)
        self._btn(btn_row, "← Back", self._show_home, color=BG_ACCENT, fg=CLR_WHITE).pack(side='left', padx=10)
        self._btn(btn_row, "Next  →", self._validate_demographics).pack(side='left', padx=10)

    def _validate_demographics(self):
        try:
            age = int(self.age_var.get())
            assert 5 <= age <= 120
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid age (5–120).")
            return
        try:
            slp = float(self.slp_var.get())
            assert 0 <= slp <= 24
        except:
            messagebox.showerror("Invalid Input", "Sleep hours must be between 0 and 24.")
            return
        try:
            exc = int(self.exc_var.get())
            assert 0 <= exc <= 7
        except:
            messagebox.showerror("Invalid Input", "Exercise days must be between 0 and 7.")
            return
        self._show_questionnaire()

    # ── Screen 3: Questionnaire ───────────────────────────────────────────────
    def _show_questionnaire(self):
        self._clear()
        self._header("Step 2 of 3 — DASS-21 Questionnaire")

        # Scrollable frame
        container = self._frame(self)
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        inner = self._frame(canvas)
        inner_window = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_resize(e):
            canvas.itemconfig(inner_window, width=e.width)
        canvas.bind('<Configure>', _on_resize)

        def _on_scroll(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_scroll)

        inner.bind('<Configure>', lambda e: canvas.configure(
            scrollregion=canvas.bbox('all')))

        # Instructions
        inst_frame = self._frame(inner, bg=BG_CARD)
        inst_frame.pack(fill='x', padx=20, pady=(10,0))
        self._label(inst_frame,
                    "Over the past week, how much did each statement apply to you?",
                    bg=BG_CARD, fg=CLR_TEAL, font=self.heading_font).pack(pady=8)
        self._label(inst_frame,
                    "0 = Not at all   1 = Sometimes   2 = Often   3 = Almost always",
                    bg=BG_CARD, fg=CLR_GRAY, font=self.small_font).pack(pady=(0,8))

        # Questions
        for i, (q_text, subscale) in enumerate(DASS21_QUESTIONS):
            qf = self._frame(inner, bg=BG_CARD if i % 2 == 0 else BG_ACCENT)
            qf.pack(fill='x', padx=20, pady=2)

            top = self._frame(qf, bg=qf['bg'])
            top.pack(fill='x', padx=16, pady=(10,4))

            badge_color = {"depression": "#7C4DFF",
                           "anxiety":    "#00B4D8",
                           "stress":     "#FF9800"}.get(subscale, CLR_GRAY)
            badge = tk.Label(top, text=f" {subscale.upper()} ", bg=badge_color,
                             fg='white', font=self.small_font, padx=4, pady=1)
            badge.pack(side='left', padx=(0,8))

            self._label(top, f"Q{i+1}. {q_text}", bg=qf['bg'], fg=CLR_WHITE,
                        font=self.body_font, wraplength=560,
                        justify='left').pack(side='left', fill='x', expand=True)

            rb_row = self._frame(qf, bg=qf['bg'])
            rb_row.pack(fill='x', padx=16, pady=(0,10))
            for val, lbl in enumerate(["0", "1", "2", "3"]):
                rb = tk.Radiobutton(rb_row, text=lbl, variable=self.q_vars[i],
                                    value=val, bg=qf['bg'], fg=CLR_WHITE,
                                    selectcolor=badge_color,
                                    activebackground=qf['bg'],
                                    font=self.body_font)
                rb.pack(side='left', padx=14)

        # Navigation
        nav = self._frame(inner)
        nav.pack(pady=20)
        self._btn(nav, "← Back", self._show_demographics,
                  color=BG_ACCENT, fg=CLR_WHITE).pack(side='left', padx=10)
        self._btn(nav, "Get Results  →", self._run_prediction).pack(side='left', padx=10)

    # ── Screen 4: Results ────────────────────────────────────────────────────
    def _run_prediction(self):
        # Validate all answered
        unanswered = [i+1 for i, v in enumerate(self.q_vars) if v.get() == -1]
        if unanswered:
            messagebox.showwarning("Incomplete",
                f"Please answer all questions.\nUnanswered: {unanswered}")
            return

        q_scores = [v.get() for v in self.q_vars]

        # Subscale totals
        dep_score = sum(q_scores[i] for i in DEPRESSION_Q)
        anx_score = sum(q_scores[i] for i in ANXIETY_Q)
        str_score = sum(q_scores[i] for i in STRESS_Q)

        # Severity labels (DASS-21 thresholds)
        def dep_level(s):
            if s <= 4: return "Normal",   CLR_GREEN
            if s <= 6: return "Mild",     "#8BC34A"
            if s <= 10: return "Moderate", CLR_ORANGE
            if s <= 13: return "Severe",   CLR_RED
            return "Extremely Severe", "#B71C1C"

        def anx_level(s):
            if s <= 3: return "Normal",   CLR_GREEN
            if s <= 5: return "Mild",     "#8BC34A"
            if s <= 7: return "Moderate", CLR_ORANGE
            if s <= 9: return "Severe",   CLR_RED
            return "Extremely Severe", "#B71C1C"

        def str_level(s):
            if s <= 7:  return "Normal",   CLR_GREEN
            if s <= 9:  return "Mild",     "#8BC34A"
            if s <= 12: return "Moderate", CLR_ORANGE
            if s <= 16: return "Severe",   CLR_RED
            return "Extremely Severe", "#B71C1C"

        d_lbl, d_clr = dep_level(dep_score)
        a_lbl, a_clr = anx_level(anx_score)
        s_lbl, s_clr = str_level(str_score)

        # ML Prediction
        prediction = None
        probability = None
        if MODEL_LOADED:
            try:
                age = float(self.age_var.get())
                gen_enc = {"Male": 1, "Female": 0, "Other": 2}.get(self.gen_var.get(), 1)
                slp = float(self.slp_var.get())
                exc = float(self.exc_var.get())
                features = [age, gen_enc, slp, exc] + q_scores
                features_arr = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_arr)
                prediction  = best_model.predict(features_scaled)[0]
                probability = best_model.predict_proba(features_scaled)[0][1]
            except Exception as e:
                prediction = None

        self._show_results(dep_score, d_lbl, d_clr,
                           anx_score, a_lbl, a_clr,
                           str_score, s_lbl, s_clr,
                           prediction, probability)

    def _show_results(self, dep_score, d_lbl, d_clr,
                             anx_score, a_lbl, a_clr,
                             str_score, s_lbl, s_clr,
                             prediction, probability):
        self._clear()
        self._header("Step 3 of 3 — Your Results")

        container = self._frame(self)
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        inner = self._frame(canvas)
        inner_window = canvas.create_window((0,0), window=inner, anchor='nw')
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(inner_window, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        # ── ML Result Banner ──
        if prediction is not None:
            risk_color = CLR_RED if prediction == 1 else CLR_GREEN
            risk_text  = ("⚠  Potential Mental Health Concern Detected"
                          if prediction == 1 else
                          "✓  No Significant Concern Detected")
            prob_text  = f"Model confidence: {probability:.0%}"
            banner = self._frame(inner, bg=risk_color)
            banner.pack(fill='x', padx=20, pady=(14,4))
            self._label(banner, risk_text, bg=risk_color, fg='white',
                        font=self.heading_font).pack(pady=(10,2))
            self._label(banner, prob_text, bg=risk_color, fg='white',
                        font=self.small_font).pack(pady=(0,10))

        # ── Subscale Cards ──
        self._label(inner, "Subscale Breakdown", fg=CLR_TEAL,
                    font=self.heading_font).pack(pady=(14,6))

        subscales = [
            ("Depression", dep_score, 21, d_lbl, d_clr,
             "Low mood, hopelessness, lack of energy"),
            ("Anxiety",    anx_score, 21, a_lbl, a_clr,
             "Physical arousal, fear, panic"),
            ("Stress",     str_score, 21, s_lbl, s_clr,
             "Tension, irritability, difficulty relaxing"),
        ]

        for name, score, max_score, level, color, desc in subscales:
            card = self._frame(inner, bg=BG_CARD)
            card.pack(fill='x', padx=40, pady=6)

            row = self._frame(card, bg=BG_CARD)
            row.pack(fill='x', padx=16, pady=(12,4))
            self._label(row, name, bg=BG_CARD, fg=CLR_WHITE,
                        font=self.heading_font).pack(side='left')
            lbl_badge = tk.Label(row, text=f" {level} ", bg=color, fg='white',
                                 font=self.small_font, padx=6, pady=2)
            lbl_badge.pack(side='right')
            self._label(row, f"Score: {score}/{max_score}",
                        bg=BG_CARD, fg=CLR_GRAY,
                        font=self.body_font).pack(side='right', padx=12)

            # Progress bar
            bar_bg = self._frame(card, bg="#2A2A4A", height=10)
            bar_bg.pack(fill='x', padx=16, pady=(0,4))
            bar_bg.pack_propagate(False)
            pct = score / max_score
            bar_fg = self._frame(bar_bg, bg=color, height=10)
            bar_fg.place(relwidth=pct, relheight=1.0)

            self._label(card, desc, bg=BG_CARD, fg=CLR_GRAY,
                        font=self.small_font).pack(padx=16, pady=(0,12), anchor='w')

        # ── Disclaimer ──
        disc = self._frame(inner, bg=BG_ACCENT)
        disc.pack(fill='x', padx=20, pady=14)
        self._label(disc,
            "⚠  IMPORTANT DISCLAIMER\n\n"
            "This screening result is NOT a medical diagnosis. It is an educational aid\n"
            "developed as a university project. If you are experiencing distress, please\n"
            "consult a licensed mental health professional or contact a crisis helpline.",
            bg=BG_ACCENT, fg=CLR_ORANGE, font=self.small_font,
            justify='center', wraplength=680).pack(pady=16)

        # ── Actions ──
        btn_row = self._frame(inner)
        btn_row.pack(pady=16)
        self._btn(btn_row, "Retake Assessment",
                  self._reset, color=BG_ACCENT, fg=CLR_WHITE).pack(side='left', padx=12)
        self._btn(btn_row, "Exit",
                  self.quit, color=CLR_RED, fg='white').pack(side='left', padx=12)

    def _reset(self):
        for v in self.q_vars:
            v.set(-1)
        self.age_var.set("")
        self.gen_var.set("Male")
        self.slp_var.set("7")
        self.exc_var.set("3")
        self._show_home()

    def _header(self, step_text):
        header = self._frame(self, bg=BG_ACCENT)
        header.pack(fill='x')
        self._label(header, "🧠  MindCheck", bg=BG_ACCENT, fg=CLR_TEAL,
                    font=self.title_font).pack(side='left', padx=20, pady=14)
        self._label(header, step_text, bg=BG_ACCENT, fg=CLR_GRAY,
                    font=self.body_font).pack(side='right', padx=20)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MindCheckApp()
    app.mainloop()
