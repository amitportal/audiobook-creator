import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import logging
import sys
from pathlib import Path
from typing import Optional
import queue

from .audiobook import AudiobookGenerator
from .tts_engine import get_tts_engine
from .chunker import TextChunker
from . import __version__

# Configure logging to write to a queue
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(record)

class AudiobookCreatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"Audiobook Creator v{__version__}")
        self.geometry("800x700")
        
        # Set theme
        ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_widgets()
        self.setup_logging()
        self.check_log_queue()

    def create_widgets(self):
        # Main container
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(1, weight=1)

        # --- File Selection ---
        self.create_section_label(0, "File Selection")
        
        # Input File
        self.input_label = ctk.CTkLabel(self.main_frame, text="Input Markdown:")
        self.input_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.input_entry = ctk.CTkEntry(self.main_frame)
        self.input_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        self.input_btn = ctk.CTkButton(self.main_frame, text="Browse", width=80, command=self.browse_input)
        self.input_btn.grid(row=1, column=2, padx=10, pady=5)

        # Output Directory
        self.output_label = ctk.CTkLabel(self.main_frame, text="Output Directory:")
        self.output_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.output_entry = ctk.CTkEntry(self.main_frame)
        self.output_entry.insert(0, "./output")
        self.output_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        self.output_btn = ctk.CTkButton(self.main_frame, text="Browse", width=80, command=self.browse_output)
        self.output_btn.grid(row=2, column=2, padx=10, pady=5)

        # --- Model Settings ---
        self.create_section_label(3, "Model Settings")

        # Model Selection
        self.model_label = ctk.CTkLabel(self.main_frame, text="TTS Model:")
        self.model_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        
        self.model_var = ctk.StringVar(value="supertonic")
        self.model_menu = ctk.CTkOptionMenu(
            self.main_frame, 
            values=["supertonic", "kokoro", "chatterbox", "soprano", "miratts"],
            variable=self.model_var
        )
        self.model_menu.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

        # Voice Style
        self.voice_label = ctk.CTkLabel(self.main_frame, text="Voice Style:")
        self.voice_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        
        self.voice_entry = ctk.CTkEntry(self.main_frame)
        self.voice_entry.insert(0, "default")
        self.voice_entry.grid(row=5, column=1, padx=10, pady=5, sticky="ew")
        
        # --- Options ---
        self.create_section_label(6, "Options")
        
        self.options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.options_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        
        self.cache_var = ctk.BooleanVar(value=False)
        self.cache_chk = ctk.CTkCheckBox(self.options_frame, text="Use Cache", variable=self.cache_var)
        self.cache_chk.pack(side="left", padx=10)
        
        self.concat_var = ctk.BooleanVar(value=False)
        self.concat_chk = ctk.CTkCheckBox(self.options_frame, text="Concatenate Output", variable=self.concat_var)
        self.concat_chk.pack(side="left", padx=10)
        
        self.dynamic_pause_var = ctk.BooleanVar(value=True)
        self.dynamic_pause_chk = ctk.CTkCheckBox(self.options_frame, text="Dynamic Pauses", variable=self.dynamic_pause_var)
        self.dynamic_pause_chk.pack(side="left", padx=10)

        # --- Actions ---
        self.start_btn = ctk.CTkButton(
            self.main_frame, 
            text="Start Generation", 
            font=("Arial", 16, "bold"),
            height=40,
            command=self.start_generation
        )
        self.start_btn.grid(row=8, column=0, columnspan=3, padx=20, pady=20, sticky="ew")

        # --- Logs ---
        self.log_label = ctk.CTkLabel(self.main_frame, text="Logs:")
        self.log_label.grid(row=9, column=0, padx=10, pady=(10,0), sticky="w")
        
        self.log_text = ctk.CTkTextbox(self.main_frame, height=200)
        self.log_text.grid(row=10, column=0, columnspan=3, padx=10, pady=5, sticky="nsew")
        self.main_frame.grid_rowconfigure(10, weight=1)

    def create_section_label(self, row, text):
        label = ctk.CTkLabel(self.main_frame, text=text, font=("Arial", 14, "bold"))
        label.grid(row=row, column=0, columnspan=3, padx=10, pady=(15, 5), sticky="w")
        separator = ctk.CTkProgressBar(self.main_frame, height=2)
        separator.set(1)
        separator.grid(row=row, column=0, columnspan=3, padx=10, pady=(35, 0), sticky="ew")

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Markdown Files", "*.md"), ("All Files", "*.*")])
        if filename:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, filename)

    def browse_output(self):
        dirname = filedialog.askdirectory()
        if dirname:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, dirname)

    def setup_logging(self):
        handler = QueueHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    def check_log_queue(self):
        while not log_queue.empty():
            record = log_queue.get()
            msg = self.format_log_record(record)
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.after(100, self.check_log_queue)

    def format_log_record(self, record):
        return f"[{record.levelname}] {record.getMessage()}"

    def start_generation(self):
        input_path = self.input_entry.get()
        output_path = self.output_entry.get()
        model = self.model_var.get()
        voice = self.voice_entry.get()
        use_cache = self.cache_var.get()
        concat = self.concat_var.get()
        dynamic_pauses = self.dynamic_pause_var.get()

        if not input_path:
            self.log_text.insert("end", "[ERROR] Please select an input file.\n")
            return

        self.start_btn.configure(state="disabled", text="Generating...")
        
        # Run in thread
        thread = threading.Thread(target=self.run_generator, args=(
            input_path, output_path, model, voice, use_cache, concat, dynamic_pauses
        ))
        thread.daemon = True
        thread.start()

    def run_generator(self, input_path, output_path, model, voice, use_cache, concat, dynamic_pauses):
        try:
            logging.info("Initializing components...")
            
            tts_engine = get_tts_engine(model_name=model, voice_style=voice)
            tts_engine.load_model()
            
            chunker = TextChunker(max_chunk_size=1000)
            
            generator = AudiobookGenerator(
                tts_engine=tts_engine,
                chunker=chunker,
                output_dir=Path(output_path),
                use_dynamic_pauses=dynamic_pauses,
                use_cache=use_cache
            )
            
            logging.info(f"Starting generation for: {input_path}")
            generator.generate_audiobook(Path(input_path), concatenate=concat)
            
            logging.info("Generation completed successfully!")
            
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            self.after(0, lambda: self.start_btn.configure(state="normal", text="Start Generation"))

def main():
    app = AudiobookCreatorApp()
    app.mainloop()

if __name__ == "__main__":
    main()
