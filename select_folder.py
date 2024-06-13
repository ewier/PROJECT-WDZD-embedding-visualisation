import tkinter as tk
from tkinter import filedialog
import json

def choose_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

if __name__ == "__main__":
    folder_path = choose_folder()
    with open("selected_folder.json", "w") as f:
        json.dump({"folder_path": folder_path}, f)
    print(f"Selected folder: {folder_path}")
