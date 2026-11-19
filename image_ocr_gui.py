"""
NeuralOCR — Intelligent Handwriting Recognition
Python / Tkinter Desktop Application

Stack  : Python · TensorFlow · OpenCV · Tkinter
Team   : Md Maskur Alam · Ragib Ahmed · Shahber Chauhan · Ismail Javed
Lab    : MIET Research Lab, Meerut  ©2026
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import sys
import io

# ── Optional heavy imports (graceful degradation if not installed) ──────────
try:
    from PIL import Image, ImageTk, ImageOps, ImageFilter
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import tensorflow as tf
    TF_OK = True
except ImportError:
    TF_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (mirrors the web app CSS variables)
# ═══════════════════════════════════════════════════════════════════════════════
C = {
    "ink":    "#0a0c12",   # deepest bg
    "deep":   "#0e1220",
    "panel":  "#12162a",
    "card":   "#161c30",
    "rim":    "#2a3050",
    "rim2":   "#1e2438",
    "azure":  "#4a90e8",
    "gold":   "#c9a96e",
    "sage":   "#6fcf97",
    "rose":   "#e87a7a",
    "text":   "#eef2ff",
    "muted":  "#6b7a9e",
    "dim":    "#3a4260",
    "white":  "#ffffff",
    "amber":  "#f59e0b",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  OCR ENGINE  (replace simulate_ocr with real model inference)
# ═══════════════════════════════════════════════════════════════════════════════
class OCREngine:
    """
    Wraps preprocessing → segmentation → CNN inference.
    If TensorFlow / OpenCV are absent, falls back to a demo string.
    """

    MODEL_PATH = "emnist_cnn_model.h5"   # place your trained model here
    LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self):
        self.model = None
        if TF_OK and os.path.exists(self.MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(self.MODEL_PATH)
            except Exception as e:
                print(f"[OCREngine] Model load failed: {e}")

    # ── public API ────────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> tuple[str, str]:
        """
        Returns (recognised_text, info_string).
        """
        if not CV2_OK or not PIL_OK:
            return self._demo_result(image_path)

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return ("ERROR", "Could not read image file.")

        chars, positions = self._segment(img_bgr)
        if not chars:
            return ("(no characters detected)", "Try a clearer image with dark ink on white background.")

        if self.model:
            batch = np.array(chars)               # (N, 28, 28, 1)
            preds = self.model.predict(batch, verbose=0)
            letters = [self.LABELS[np.argmax(p)] for p in preds]
        else:
            # Demo: return placeholder text
            letters = ["?"] * len(chars)

        text = "".join(letters)
        info = f"{len(letters)} characters recognised · {os.path.basename(image_path)}"
        return text, info

    # ── internal helpers ──────────────────────────────────────────────────────
    def _preprocess(self, img_bgr):
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((2, 2), np.uint8)
        clean  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return clean

    def _segment(self, img_bgr):
        clean   = self._preprocess(img_bgr)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(clean)
        chars, positions = [], []

        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 40 or w > img_bgr.shape[1] * 0.9:
                continue
            roi = clean[y:y+h, x:x+w]
            # EMNIST fix: rotate 90° CW then flip horizontally
            roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            roi = cv2.flip(roi, 1)
            roi = cv2.resize(roi, (28, 28)).astype("float32") / 255.0
            roi = roi.reshape(28, 28, 1)
            chars.append(roi)
            positions.append(x)          # track x for left→right ordering

        # sort characters left-to-right
        paired = sorted(zip(positions, chars), key=lambda p: p[0])
        chars  = [c for _, c in paired]
        return chars, positions

    def _demo_result(self, image_path):
        basename = os.path.basename(image_path)
        name, _  = os.path.splitext(basename)
        demo_text = name.upper().replace("_", " ")[:20] or "HELLO WORLD"
        info = f"{len(demo_text)} characters recognised · {basename} (demo mode)"
        return demo_text, info


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLED WIDGETS  (thin helpers for consistent look)
# ═══════════════════════════════════════════════════════════════════════════════
def styled_label(parent, text="", size=12, weight="normal", color=None,
                 font_family="Courier", **kw):
    color = color or C["text"]
    return tk.Label(
        parent, text=text,
        font=(font_family, size, weight),
        fg=color, bg=kw.pop("bg", C["ink"]),
        **kw
    )


def styled_btn(parent, text, command, bg=C["azure"], fg=C["white"],
               size=10, pad=(10, 6), relief=tk.FLAT, **kw):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
        font=("Courier", size, "bold"),
        relief=relief, bd=0,
        padx=pad[0], pady=pad[1],
        cursor="hand2",
        **kw
    )
    # hover effects
    def on_enter(e):
        btn.config(bg=_lighten(bg))
    def on_leave(e):
        btn.config(bg=bg)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


def _lighten(hex_color, amount=30):
    """Lighten a hex colour by `amount` (0–255)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = min(255, r + amount)
    g = min(255, g + amount)
    b = min(255, b + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
class NeuralOCRApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NeuralOCR — Intelligent Handwriting Recognition")
        self.geometry("1100x700")
        self.minsize(900, 600)
        self.configure(bg=C["ink"])
        self.resizable(True, True)

        self._engine     = OCREngine()
        self._last_result   = ""
        self._last_filepath = ""
        self._preview_img   = None   # keep PhotoImage reference

        self._build_ui()
        self._animate_status_dot()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI BUILD
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_nav()
        main = tk.Frame(self, bg=C["ink"])
        main.pack(fill=tk.BOTH, expand=True, padx=24, pady=(12, 24))
        self._build_app_card(main)
        self._build_strip(main)
        self._build_footer()

    # ── NAV BAR ──────────────────────────────────────────────────────────────
    def _build_nav(self):
        nav = tk.Frame(self, bg=C["deep"], height=56)
        nav.pack(fill=tk.X)
        nav.pack_propagate(False)

        # Logo box
        logo_frame = tk.Frame(nav, bg="#4a90e8", width=32, height=32)
        logo_frame.place(x=20, rely=0.5, anchor="w")
        logo_frame.pack_propagate(False)
        tk.Label(logo_frame, text="N", bg="#4a90e8", fg=C["white"],
                 font=("Georgia", 14, "bold")).place(relx=0.5, rely=0.5, anchor="center")

        wordmark = tk.Frame(nav, bg=C["deep"])
        wordmark.place(x=60, rely=0.5, anchor="w")
        tk.Label(wordmark, text="Neural", bg=C["deep"], fg=C["text"],
                 font=("Georgia", 14, "bold")).pack(side=tk.LEFT)
        tk.Label(wordmark, text="OCR", bg=C["deep"], fg=C["azure"],
                 font=("Georgia", 14, "bold")).pack(side=tk.LEFT)

        # Nav links (right side)
        links = tk.Frame(nav, bg=C["deep"])
        links.place(relx=0.65, rely=0.5, anchor="w")
        for name in ["Product", "Technology", "Research", "Docs"]:
            lbl = tk.Label(links, text=name.upper(), bg=C["deep"], fg=C["muted"],
                           font=("Courier", 9), cursor="hand2")
            lbl.pack(side=tk.LEFT, padx=14)
            lbl.bind("<Enter>", lambda e, l=lbl: l.config(fg=C["text"]))
            lbl.bind("<Leave>", lambda e, l=lbl: l.config(fg=C["muted"]))

        # CTA button
        cta = styled_btn(nav, "GET EARLY ACCESS", lambda: None,
                         bg=C["deep"], fg=C["text"], size=8, pad=(10, 5))
        cta.config(relief=tk.SOLID, bd=1, highlightbackground=C["rim"])
        cta.place(relx=0.98, rely=0.5, anchor="e")

    # ── APP CARD ─────────────────────────────────────────────────────────────
    def _build_app_card(self, parent):
        card = tk.Frame(parent, bg=C["card"], bd=1, relief=tk.SOLID,
                        highlightbackground=C["rim"], highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        inner = tk.Frame(card, bg=C["card"])
        inner.pack(fill=tk.BOTH, expand=True)

        self._build_left_panel(inner)
        sep = tk.Frame(inner, bg=C["rim2"], width=1)
        sep.pack(side=tk.LEFT, fill=tk.Y)
        self._build_right_panel(inner)

    # ── LEFT PANEL ───────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=C["panel"], width=300)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        # Header
        hdr = tk.Frame(left, bg=C["panel"])
        hdr.pack(fill=tk.X, padx=16, pady=(16, 10))
        styled_label(hdr, "INPUT", size=8, color=C["muted"], bg=C["panel"],
                     font_family="Courier").pack(anchor="w")
        styled_label(hdr, "Drop a handwritten image or browse", size=10,
                     color=C["text"], bg=C["panel"]).pack(anchor="w", pady=(4, 0))

        # Drop zone
        self._drop_frame = tk.Frame(left, bg=C["ink"], bd=2, relief=tk.GROOVE,
                                    highlightbackground=C["dim"], highlightthickness=1)
        self._drop_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))

        self._drop_glyph = styled_label(self._drop_frame, "⬆", size=22,
                                        color=C["muted"], bg=C["ink"])
        self._drop_glyph.pack(pady=(32, 4))
        styled_label(self._drop_frame, "Drag & drop image here", size=11,
                     color=C["text"], bg=C["ink"]).pack()
        styled_label(self._drop_frame, "or click to browse", size=9,
                     color=C["muted"], bg=C["ink"]).pack()
        styled_label(self._drop_frame, "PNG · JPG · BMP · TIFF", size=8,
                     color=C["dim"], bg=C["ink"]).pack(pady=(4, 0))

        self._preview_label = tk.Label(self._drop_frame, bg=C["ink"])
        self._preview_label.pack(pady=(8, 0))

        self._drop_success = styled_label(self._drop_frame, "", size=9,
                                          color=C["sage"], bg=C["ink"])
        self._drop_success.pack()

        # Bind drop-zone clicks
        for w in [self._drop_frame, self._drop_glyph, self._drop_success]:
            w.bind("<Button-1>", lambda e: self._browse_file())

        # Drag-and-drop bindings (Linux/Windows)
        self._drop_frame.drop_target_register = getattr(self._drop_frame, "drop_target_register", None)
        self._setup_dnd()

        # Browse button
        styled_btn(left, "  Browse & Upload  ", self._browse_file,
                   bg=C["azure"], size=10, pad=(12, 8)).pack(
            fill=tk.X, padx=14, pady=(0, 4))

        styled_label(left, "Supported: PNG · JPG · BMP · TIFF", size=8,
                     color=C["dim"], bg=C["panel"]).pack()

        # Export section
        exp_frame = tk.Frame(left, bg=C["deep"])
        exp_frame.pack(fill=tk.X, padx=0, pady=(8, 0))
        tk.Frame(exp_frame, bg=C["rim2"], height=1).pack(fill=tk.X)
        inner_exp = tk.Frame(exp_frame, bg=C["deep"])
        inner_exp.pack(fill=tk.X, padx=14, pady=10)

        styled_label(inner_exp, "EXPORT RESULTS", size=8,
                     color=C["dim"], bg=C["deep"], font_family="Courier").pack(anchor="w")

        btn_row = tk.Frame(inner_exp, bg=C["deep"])
        btn_row.pack(fill=tk.X, pady=(6, 0))

        self._save_txt_btn = styled_btn(btn_row, "↓ Save Text", self._save_txt,
                                        bg=C["rim2"], fg=C["muted"], size=9, pad=(8, 5))
        self._save_txt_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self._save_txt_btn.config(state=tk.DISABLED)

        self._save_img_btn = styled_btn(btn_row, "↓ Save Image", self._save_img,
                                        bg=C["rim2"], fg=C["muted"], size=9, pad=(8, 5))
        self._save_img_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self._save_img_btn.config(state=tk.DISABLED)

        styled_btn(inner_exp, "⎘  Copy to Clipboard", self._copy_clipboard,
                   bg=C["rim2"], fg=C["muted"], size=9, pad=(8, 5)).pack(
            fill=tk.X, pady=(6, 0))

    # ── RIGHT PANEL ──────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=C["ink"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tab bar
        tab_bar = tk.Frame(right, bg=C["deep"], height=46)
        tab_bar.pack(fill=tk.X)
        tab_bar.pack_propagate(False)

        self._tab_frames   = {}
        self._tab_buttons  = {}
        self._active_tab   = tk.StringVar(value="overview")

        tab_container = tk.Frame(tab_bar, bg=C["deep"])
        tab_container.pack(side=tk.LEFT, fill=tk.Y, padx=16)

        for key, label in [("overview", "Overview"), ("about", "About"), ("team", "Team")]:
            btn = tk.Label(
                tab_container, text=label.upper(),
                font=("Courier", 9), fg=C["dim"],
                bg=C["deep"], cursor="hand2", padx=14, pady=14
            )
            btn.pack(side=tk.LEFT)
            btn.bind("<Button-1>", lambda e, k=key: self._switch_tab(k))
            self._tab_buttons[key] = btn

        # Status badge (right side of tab bar)
        status_frame = tk.Frame(tab_bar, bg=C["deep"])
        status_frame.pack(side=tk.RIGHT, padx=16, pady=10)
        self._status_dot = tk.Label(status_frame, text="●", font=("Courier", 8),
                                     fg=C["dim"], bg=C["deep"])
        self._status_dot.pack(side=tk.LEFT)
        self._status_lbl = tk.Label(status_frame, text="READY",
                                     font=("Courier", 8), fg=C["dim"], bg=C["deep"])
        self._status_lbl.pack(side=tk.LEFT, padx=(4, 0))

        # Content area
        content_area = tk.Frame(right, bg=C["ink"])
        content_area.pack(fill=tk.BOTH, expand=True)

        self._tab_frames["overview"] = self._build_overview_tab(content_area)
        self._tab_frames["about"]    = self._build_about_tab(content_area)
        self._tab_frames["team"]     = self._build_team_tab(content_area)

        self._switch_tab("overview")

    def _switch_tab(self, key):
        self._active_tab.set(key)
        for k, frame in self._tab_frames.items():
            if k == key:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()
        for k, btn in self._tab_buttons.items():
            if k == key:
                btn.config(fg=C["azure"])
            else:
                btn.config(fg=C["dim"])

    # ── OVERVIEW TAB ─────────────────────────────────────────────────────────
    def _build_overview_tab(self, parent):
        frame = tk.Frame(parent, bg=C["ink"])

        # Feature grid (2×2)
        grid = tk.Frame(frame, bg=C["ink"])
        grid.pack(fill=tk.X, padx=20, pady=16)

        features = [
            ("⬡", "Preprocessing",  C["azure"], "OpenCV binarisation, adaptive thresholding, and noise removal."),
            ("◎", "Segmentation",   C["gold"],  "Connected component analysis isolates each glyph."),
            ("◈", "CNN Inference",  C["sage"],  "EMNIST-trained CNN classifies all 26 uppercase characters."),
            ("✦", "Batch Output",   C["rose"],  "All characters predicted in a single forward pass."),
        ]

        for i, (icon, title, color, desc) in enumerate(features):
            row, col = divmod(i, 2)
            card = tk.Frame(grid, bg=C["card"], bd=1, relief=tk.FLAT,
                            highlightbackground=C["rim2"], highlightthickness=1)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            grid.columnconfigure(col, weight=1)

            icon_lbl = tk.Label(card, text=icon, font=("Courier", 18),
                                fg=color, bg=C["card"])
            icon_lbl.pack(anchor="w", padx=14, pady=(14, 4))
            tk.Label(card, text=title, font=("Georgia", 12, "bold"),
                     fg=C["text"], bg=C["card"]).pack(anchor="w", padx=14)
            tk.Label(card, text=desc, font=("Courier", 9), fg=C["muted"],
                     bg=C["card"], wraplength=180, justify=tk.LEFT).pack(
                anchor="w", padx=14, pady=(4, 14))

        # Result card
        res_frame = tk.Frame(frame, bg=C["ink"])
        res_frame.pack(fill=tk.X, padx=20, pady=(0, 16))
        styled_label(res_frame, "RECOGNISED OUTPUT", size=8, color=C["dim"],
                     bg=C["ink"], font_family="Courier").pack(anchor="w", pady=(0, 6))

        res_card = tk.Frame(res_frame, bg="#0d1020", bd=0,
                            highlightbackground=C["rim2"], highlightthickness=1)
        res_card.pack(fill=tk.X)

        # Accent bar
        tk.Frame(res_card, bg=C["azure"], width=4).pack(side=tk.LEFT, fill=tk.Y)

        body = tk.Frame(res_card, bg="#0d1020")
        body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=14, pady=12)

        self._result_text = styled_label(body, "—", size=20,
                                         color=C["text"], bg="#0d1020",
                                         font_family="Georgia")
        self._result_text.pack(anchor="w")
        self._result_meta = styled_label(body, "Upload an image to begin recognition",
                                         size=9, color=C["dim"], bg="#0d1020",
                                         font_family="Courier")
        self._result_meta.pack(anchor="w", pady=(4, 0))

        return frame

    # ── ABOUT TAB ────────────────────────────────────────────────────────────
    def _build_about_tab(self, parent):
        frame = tk.Frame(parent, bg=C["ink"])

        canvas = tk.Canvas(frame, bg=C["ink"], bd=0, highlightthickness=0)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=C["ink"])
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        about_items = [
            ("Image Preprocessing",
             "Raw images are binarised using OpenCV adaptive thresholding. "
             "Noise removal and morphological operations ensure clean input for segmentation."),
            ("Character Segmentation",
             "Connected component analysis detects and isolates individual character blobs. "
             "Each glyph is extracted and resized to 28×28 pixels for the classifier."),
            ("CNN Architecture",
             "A convolutional neural network trained on the EMNIST dataset classifies all 26 "
             "uppercase English letters. EMNIST orientation corrections (rotate + flip) are "
             "applied before inference."),
            ("Batch Inference & Export",
             "All extracted characters are batched into a single prediction call for efficiency. "
             "Results can be exported as plain text files, image copies, or copied to the clipboard."),
        ]

        for title, body in about_items:
            card = tk.Frame(inner, bg=C["card"], bd=1, relief=tk.FLAT,
                            highlightbackground=C["rim2"], highlightthickness=1)
            card.pack(fill=tk.X, padx=20, pady=6)
            tk.Label(card, text=title, font=("Georgia", 12, "bold"),
                     fg=C["azure"], bg=C["card"]).pack(anchor="w", padx=14, pady=(12, 4))
            tk.Label(card, text=body, font=("Courier", 9), fg=C["muted"],
                     bg=C["card"], wraplength=480, justify=tk.LEFT).pack(
                anchor="w", padx=14, pady=(0, 12))

        return frame

    # ── TEAM TAB ─────────────────────────────────────────────────────────────
    def _build_team_tab(self, parent):
        frame = tk.Frame(parent, bg=C["ink"])

        rows = [
            ("Lead Developer",  "Md Maskur Alam"),
            ("Programme",       "B.Tech CSE — 2022–2026"),
            ("Institution",     "MIET, Meerut"),
            ("Email",           "your-email@example.com"),
            ("Team",            "Ragib Ahmed"),
            ("",                "Shahber Chauhan"),
            ("",                "Ismail Javed"),
            ("Project Type",    "Final Year Research Project"),
            ("Stack",           "Python · TensorFlow · OpenCV · Tkinter"),
        ]

        table = tk.Frame(frame, bg=C["ink"])
        table.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

        for i, (key, val) in enumerate(rows):
            bg = C["card"] if i % 2 == 0 else C["ink"]
            row_frame = tk.Frame(table, bg=bg)
            row_frame.pack(fill=tk.X)
            tk.Label(row_frame, text=key, font=("Courier", 9), fg=C["muted"],
                     bg=bg, width=18, anchor="w").pack(
                side=tk.LEFT, padx=14, pady=10)
            tk.Label(row_frame, text=val, font=("Courier", 10), fg=C["text"],
                     bg=bg, anchor="w").pack(side=tk.LEFT, pady=10)

        return frame

    # ── BOTTOM STRIP ─────────────────────────────────────────────────────────
    def _build_strip(self, parent):
        strip = tk.Frame(parent, bg=C["ink"])
        strip.pack(fill=tk.X, pady=(10, 0))

        steps = [
            ("01 / UPLOAD",  "Drop or Browse",    "Drag any PNG, JPG, BMP, or TIFF image onto the upload zone — no configuration needed."),
            ("02 / PROCESS", "CNN Recognition",   "Our CNN preprocesses, segments, and classifies each character in one optimised batch pass."),
            ("03 / EXPORT",  "Save & Share",      "Download as .txt, save the image, or copy the output directly to your clipboard."),
        ]

        for num, title, body in steps:
            card = tk.Frame(strip, bg=C["card"], bd=1, relief=tk.FLAT,
                            highlightbackground=C["rim2"], highlightthickness=1)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
            tk.Label(card, text=num, font=("Courier", 9), fg=C["azure"],
                     bg=C["card"]).pack(anchor="w", padx=16, pady=(14, 2))
            tk.Label(card, text=title, font=("Georgia", 13, "bold"),
                     fg=C["text"], bg=C["card"]).pack(anchor="w", padx=16)
            tk.Label(card, text=body, font=("Courier", 9), fg=C["muted"],
                     bg=C["card"], wraplength=260, justify=tk.LEFT).pack(
                anchor="w", padx=16, pady=(4, 14))

    # ── FOOTER ───────────────────────────────────────────────────────────────
    def _build_footer(self):
        footer = tk.Frame(self, bg=C["deep"], height=40)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)

        tk.Label(footer, text="Neural", font=("Georgia", 11, "bold"),
                 fg=C["muted"], bg=C["deep"]).place(x=20, rely=0.5, anchor="w")
        tk.Label(footer, text="OCR", font=("Georgia", 11, "bold"),
                 fg=C["azure"], bg=C["deep"]).place(x=62, rely=0.5, anchor="w")

        tk.Label(footer, text="© 2026 MIET Research Lab, Meerut",
                 font=("Courier", 8), fg=C["dim"], bg=C["deep"]).place(
            relx=0.5, rely=0.5, anchor="center")

        tk.Label(footer, text="Md Maskur Alam · Ragib Ahmed · Shahber Chauhan · Ismail Javed",
                 font=("Courier", 7), fg=C["dim"], bg=C["deep"]).place(
            relx=0.98, rely=0.5, anchor="e")

    # ─────────────────────────────────────────────────────────────────────────
    #  STATUS DOT ANIMATION
    # ─────────────────────────────────────────────────────────────────────────
    def _animate_status_dot(self):
        self._dot_phase = 0

        def tick():
            if not self.winfo_exists():
                return
            current = self._status_lbl.cget("text")
            if current == "READY":
                alpha = int(128 + 127 * (self._dot_phase % 1.0 < 0.5))
                col = f"#{alpha:02x}{alpha:02x}{alpha+30 if alpha+30<256 else 255:02x}"
                self._status_dot.config(fg=C["dim"])
            self._dot_phase += 0.05
            self.after(80, tick)

        tick()

    def _set_status(self, state: str):
        states = {
            "ready":   ("READY",       C["dim"]),
            "working": ("PROCESSING…", C["amber"]),
            "done":    ("DONE",        C["sage"]),
            "error":   ("ERROR",       C["rose"]),
        }
        label, color = states.get(state, ("READY", C["dim"]))
        self._status_dot.config(fg=color)
        self._status_lbl.config(text=label, fg=color)

    # ─────────────────────────────────────────────────────────────────────────
    #  FILE HANDLING
    # ─────────────────────────────────────────────────────────────────────────
    def _setup_dnd(self):
        """
        Attempt to register tkinterdnd2 drag-and-drop.
        Silently skips if the library is not installed.
        """
        try:
            from tkinterdnd2 import DND_FILES
            self._drop_frame.drop_target_register(DND_FILES)
            self._drop_frame.dnd_bind("<<Drop>>", self._on_dnd_drop)
        except Exception:
            pass  # DND unavailable — user can use Browse button

    def _on_dnd_drop(self, event):
        path = event.data.strip().strip("{}")
        if os.path.isfile(path):
            self._process_file(path)

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Handwritten Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files",   "*.*"),
            ]
        )
        if path:
            self._process_file(path)

    def _process_file(self, path: str):
        self._last_filepath = path
        self._show_preview(path)
        self._set_status("working")
        self._result_text.config(text="—")
        self._result_meta.config(text="Running inference…")
        self._switch_tab("overview")

        # Run OCR in background thread so UI stays responsive
        t = threading.Thread(target=self._run_ocr, args=(path,), daemon=True)
        t.start()

    def _show_preview(self, path: str):
        if not PIL_OK:
            self._drop_success.config(text=f"✓  {os.path.basename(path)}")
            return
        try:
            img = Image.open(path)
            img.thumbnail((220, 130), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._preview_label.config(image=photo)
            self._preview_img = photo   # prevent GC
            # Hide placeholder widgets
            self._drop_glyph.pack_forget()
        except Exception:
            pass
        self._drop_success.config(text=f"✓  {os.path.basename(path)}")

    def _run_ocr(self, path: str):
        time.sleep(0.3)   # brief pause for UX
        try:
            text, info = self._engine.predict(path)
            self._last_result = text
            self.after(0, self._on_ocr_done, text, info)
        except Exception as e:
            self.after(0, self._on_ocr_error, str(e))

    def _on_ocr_done(self, text: str, info: str):
        self._result_text.config(text=text)
        self._result_meta.config(text=info)
        self._set_status("done")
        self._save_txt_btn.config(state=tk.NORMAL, bg=C["rim"], fg=C["text"])
        self._save_img_btn.config(state=tk.NORMAL, bg=C["rim"], fg=C["text"])

    def _on_ocr_error(self, msg: str):
        self._result_text.config(text="ERROR")
        self._result_meta.config(text=msg)
        self._set_status("error")

    # ─────────────────────────────────────────────────────────────────────────
    #  EXPORT ACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    def _save_txt(self):
        if not self._last_result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")],
            initialfile=f"ocr_result_{int(time.time())}.txt"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._last_result)
            messagebox.showinfo("Saved", f"Text saved to:\n{path}")

    def _save_img(self):
        if not self._last_filepath:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialfile=f"ocr_image_{int(time.time())}.png"
        )
        if path and PIL_OK:
            try:
                Image.open(self._last_filepath).save(path)
                messagebox.showinfo("Saved", f"Image saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _copy_clipboard(self):
        if not self._last_result:
            return
        self.clipboard_clear()
        self.clipboard_append(self._last_result)
        messagebox.showinfo("Copied", "Result copied to clipboard!")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = NeuralOCRApp()
    app.mainloop()