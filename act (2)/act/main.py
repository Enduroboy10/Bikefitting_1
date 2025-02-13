import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import ohnegui as og

# Farben & Fonts für das UI
BG_COLOR = "#2C3E50"
FG_COLOR = "#ECF0F1"
BUTTON_COLOR = "#3498DB"
BUTTON_HOVER = "#2980B9"
FONT = ("Arial", 14)

# Hauptfenster konfigurieren
root = tk.Tk()
root.title("Bikefitting App")
root.geometry("1200x800")
root.minsize(800, 600)
root.configure(bg=BG_COLOR)

# Globale Variablen
video_label = None
cap = None
is_playing = False
file_path = None

# Funktion für Vollbildmodus
def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", not root.attributes("-fullscreen"))

def close_app(event=None):
    global cap
    if cap:
        cap.release()
    root.destroy()

# Funktion zum Importieren eines Videos
def import_video():
    global cap, file_path, is_playing
    file_path = filedialog.askopenfilename(
        title="Wähle eine Videodatei",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("Alle Dateien", "*.*")]
    )
    if file_path:
        label.config(text=f"Video geladen: {os.path.basename(file_path)}")
        cap = cv2.VideoCapture(file_path)
        is_playing = True
        play_video()
    else:
        label.config(text="Kein Video ausgewählt!")

# Funktion zum Anpassen der Frame-Größe
def resize_frame(frame, max_width, max_height):
    h, w, _ = frame.shape
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))

# Funktion zum Abspielen des Videos
def play_video():
    global cap, video_label, is_playing
    if not cap or not is_playing:
        return

    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_frame(frame, 800, 450)
        img = ImageTk.PhotoImage(Image.fromarray(frame))

        if video_label is None:
            video_label = tk.Label(frame_display, bg=BG_COLOR)
            video_label.pack(expand=True)
        video_label.config(image=img)
        video_label.image = img
        root.after(30, play_video)
    else:
        cap.release()
        cap = None
        is_playing = False
        label.config(text="Videowiedergabe beendet.")

# Funktion zum Analysieren des Videos
def analyze_video():
    global cap, video_label, is_playing
    if file_path:
        if cap:
            is_playing = False
            cap.release()
            cap = None
        if video_label:
            video_label.destroy()
            video_label = None

        frames = og.extract_fr(file_path)
        resp = og.get_interf(frames[0])
        frame, angle = og.show_inference_result_with_keypoints(resp, frames[0])

        if angle > 30:
            angle_label.config(text="Sattelhöhe senken", fg="red")
        elif 25 <= angle <= 30:
            angle_label.config(text="Sattelhöhe ist optimal", fg="green")
        else:
            angle_label.config(text="Sattelhöhe erhöhen", fg="blue")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_frame(frame, 800, 450)
        img = ImageTk.PhotoImage(Image.fromarray(frame))

        video_label = tk.Label(frame_display, image=img, bg=BG_COLOR)
        video_label.image = img
        video_label.pack(expand=True)
        label.config(text="Analyse abgeschlossen.")
    else:
        label.config(text="Kein Video zum Analysieren!")

# Funktion zum Zurücksetzen
def restart_process():
    global cap, video_label, is_playing, file_path
    if cap:
        is_playing = False
        cap.release()
        cap = None
    if video_label:
        video_label.destroy()
        video_label = None

    label.config(text="Bereit für neue Analyse")
    angle_label.config(text="", fg=FG_COLOR)
    file_path = None

# UI-Gestaltung mit Frames
top_frame = tk.Frame(root, bg=BG_COLOR)
top_frame.pack(fill="x", padx=20, pady=10)

frame_display = tk.Frame(root, bg=BG_COLOR, width=800, height=450)
frame_display.pack(expand=True)

bottom_frame = tk.Frame(root, bg=BG_COLOR)
bottom_frame.pack(fill="x", padx=20, pady=20)

# Labels
label = tk.Label(top_frame, text="Willkommen zur Bikefitting App", font=("Arial", 16), bg=BG_COLOR, fg=FG_COLOR)
label.pack(pady=5)

angle_label = tk.Label(bottom_frame, text="", font=("Arial", 16), bg=BG_COLOR, fg=FG_COLOR)
angle_label.pack(pady=10)

# Buttons mit klar sichtbarem Text
import_button = tk.Button(top_frame, text="📂 Video importieren", font=FONT, fg="black", bg=BUTTON_COLOR, command=import_video)
import_button.pack(side="left", padx=10)

process_button = tk.Button(top_frame, text="🔍 Analyse starten", font=FONT, fg="black", bg=BUTTON_COLOR, command=analyze_video)
process_button.pack(side="left", padx=10)

restart_button = tk.Button(top_frame, text="🔄 Neustarten", font=FONT, fg="black", bg=BUTTON_COLOR, command=restart_process)
restart_button.pack(side="left", padx=10)

exit_button = tk.Button(top_frame, text="❌ Beenden", font=FONT, fg="black", bg=BUTTON_COLOR, command=close_app)
exit_button.pack(side="right", padx=10)

# Hotkeys für Vollbild & Beenden
root.bind("<F11>", toggle_fullscreen)
root.bind("<Escape>", close_app)

root.mainloop()
