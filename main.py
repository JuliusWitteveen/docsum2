# -------------------------------------------------------------------
# main.py
# -------------------------------------------------------------------
"""
The main module of the Document Summarizer application. This module integrates the functionalities of file handling, 
language processing, and summarization into a user-friendly GUI. It is responsible for the main application flow, 
including file selection, language detection, summarization, and saving the summarized document. It relies on the 
'file_handler', 'language_processing', and 'summarization' modules, as well as configuration settings from 'config'.
"""

import file_handler
import language_processing
import summarization
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import config  # Import the config module

# Global variables
selected_file_path = None  # holds the path of the file selected by the user for summarization.
progress = None  # refers to the progress bar widget in the GUI, showing the summarization progress.
custom_prompt_area = None  # is a text area in the GUI for customizing the summarization prompt.
chunk_size = None  # user-configurable settings for text chunking in the summarization process.
chunk_overlap = None
use_clustering = None  # BooleanVar for clustering option in the GUI.

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import configuration settings from config module for default prompts, chunk size, and overlap.
# These settings are used throughout the application to control various aspects of the summarization process.
default_prompt_en = config.DEFAULT_PROMPT_EN
DEFAULT_CHUNK_SIZE = config.DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = config.DEFAULT_CHUNK_OVERLAP

# Helper Functions
def get_api_key(file_path=r'C:\\api_key.txt'):
    """
    Retrieve the API key from a specified file.

    Parameters:
    file_path (str): The path to the API key file. Default is 'C:\\api_key.txt'.

    Returns:
    str: The API key as a string, or None if the file is not found or an error occurs.

    Exceptions:
    FileNotFoundError: If the API key file is not found at the specified path.
    IOError: If there is an error reading the API key file.
    """
    logging.info("Retrieving API key.")
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error(f"API key file not found at {file_path}")
        return None
    except IOError as e:
        logging.error(f"Error reading the API key file: {e}")
        return None

def select_file():
    """
    Opens a file dialog for the user to select a document for summarization.
    
    Returns:
    str: The file path of the selected document. Supported formats include PDF, DOCX, RTF, and TXT.
    """
    file_path = filedialog.askopenfilename(
        title="Select a Document",
        filetypes=[("PDF Files", "*.pdf"), ("Word Documents", "*.docx"), ("RTF Files", "*.rtf"), ("Text Files", "*.txt")])
    return file_path

def get_summary_prompt(file_path, api_key):
    """
    Generate a summary prompt based on the language of the text in the given file.

    Parameters:
    file_path (str): The path of the file to summarize.
    api_key (str): The API key for any external services used.

    Returns:
    str: A summary prompt in the detected language or the default prompt if language detection fails.

    Description:
    This function loads the document text using 'file_handler', detects its language,
    and then either translates the default English prompt to the detected language or returns
    the default prompt if translation is not required or possible.
    """
    text = file_handler.load_document(file_path)
    if not text:
        return None

    language = language_processing.detect_language(text)
    if language == "nl":
        translated_prompt = language_processing.translate_prompt(default_prompt_en, language)
        return translated_prompt
    elif language == "en":
        return default_prompt_en

    return default_prompt_en

# Background Summarization Function
def start_summarization_thread(root):
    """
    Start the summarization process in a separate thread to keep the GUI responsive.

    Parameters:
    root (tk.Tk): The root window of the application.

    Description:
    This function creates a new thread to run the summarization process, allowing the GUI
    to remain responsive and update the progress bar during the summarization.
    """
    summarization_thread = threading.Thread(target=start_summarization, args=(root,))
    summarization_thread.start()

def start_summarization(root):
    global selected_file_path, custom_prompt_area, chunk_size, chunk_overlap, use_clustering
    api_key = get_api_key()

    logging.info(f"Starting summarization. Clustering option: {use_clustering.get()}")

    if api_key and selected_file_path:
        try:
            custom_prompt_text = get_summary_prompt(selected_file_path, api_key)
            update_progress_bar(10, root)

            text = file_handler.load_document(selected_file_path)
            update_progress_bar(20, root)

            user_chunk_size = int(chunk_size.get() or DEFAULT_CHUNK_SIZE)
            user_chunk_overlap = int(chunk_overlap.get() or DEFAULT_CHUNK_OVERLAP)

            logging.info(f"Summarization parameters - Chunk size: {user_chunk_size}, Chunk overlap: {user_chunk_overlap}, Clustering: {use_clustering.get()}")

            summary = summarization.execute_summary(
                text, 
                api_key, 
                custom_prompt_text,
                user_chunk_size,
                user_chunk_overlap,
                use_clustering=use_clustering.get(),
                progress_update_callback=lambda value: update_progress_bar(value, root)
            )

            if summary:
                filename_without_ext = os.path.splitext(os.path.basename(selected_file_path))[0]
                root.after(0, lambda: save_summary_file(summary, filename_without_ext))
                update_progress_bar(100, root)

        except Exception as e:
            logging.error(f"Error in summarization process: {e}")
            messagebox.showerror("Summarization Error", f"An error occurred during summarization: {e}")
            update_progress_bar(0, root)
    else:
        messagebox.showinfo("API Key Missing", "API key is missing or invalid.")
        update_progress_bar(0, root)

def update_progress_bar(value, root):
    """
    Updates the progress bar in the GUI.

    Parameters:
    value (int): The value to set the progress bar to.
    root (tk.Tk): The root window of the application.

    This function is used throughout the summarization process to provide feedback on its progress to the user.
    """
    def set_progress(value):
        progress['value'] = value
    root.after(0, lambda: set_progress(value))

def save_summary_file(summary, filename_without_ext):
    default_summary_filename = f"{filename_without_ext}_sum"
    file_path = filedialog.asksaveasfilename(
        initialfile=default_summary_filename,
        filetypes=[("Text Files", "*.txt"), ("Word Documents", "*.docx"), ("PDF Files", "*.pdf")],
        defaultextension=".txt"
    )
    if file_path:
        # Log the first 500 characters of the summary being saved
        logging.info(f"Saving Summary (first 500 characters): {summary[:500]}...")

        file_handler.save_summary(summary, file_path)
        messagebox.showinfo("Success", f"Summary saved successfully to {file_path}")
    else:
        messagebox.showerror("Error", "No file path selected.")

# GUI Code Block
def main_gui():
    global selected_file_path, progress, custom_prompt_area, chunk_size, chunk_overlap, use_clustering

    logging.info("Initializing GUI for the Document Summarizer.")

    root = tk.Tk()
    root.title("Document Summarizer")
    root.geometry('800x600')  # Set a fixed size window for consistent layout

    primary_color = "#2E3F4F"
    secondary_color = "#4F5D75"
    text_color = "#E0FBFC"
    button_color = "#3F88C5"
    larger_font = ('Helvetica', 12)
    button_font = ('Helvetica', 10, 'bold')

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TCheckbutton', background=primary_color, foreground=text_color, font=larger_font)
    style.configure('W.TButton', font=button_font, background=button_color, foreground=text_color)
    style.map('W.TButton', background=[('active', secondary_color)], foreground=[('active', text_color)])

    root.configure(bg=primary_color)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=3)  # Adjust the weight for a wider second column

    progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
    progress.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky='ew')

    prompt_label = tk.Label(root, text="Customize the summarization prompt:", fg=text_color, bg=primary_color, font=larger_font)
    prompt_label.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky='nw')
    
    custom_prompt_area = tk.Text(root, height=15, width=50, wrap="word", bd=2, font=larger_font)
    custom_prompt_area.grid(row=2, column=0, columnspan=2, pady=10, padx=10, sticky='nsew')

    chunk_size = tk.StringVar(value=str(DEFAULT_CHUNK_SIZE))
    chunk_overlap = tk.StringVar(value=str(DEFAULT_CHUNK_OVERLAP))

    chunk_size_label = tk.Label(root, text="Chunk Size:", fg=text_color, bg=primary_color, font=larger_font)
    chunk_size_label.grid(row=3, column=0, sticky='e')
    chunk_size_entry = tk.Entry(root, textvariable=chunk_size, bd=2, font=larger_font)
    chunk_size_entry.grid(row=3, column=1, sticky='w')

    chunk_overlap_label = tk.Label(root, text="Chunk Overlap:", fg=text_color, bg=primary_color, font=larger_font)
    chunk_overlap_label.grid(row=4, column=0, sticky='e')
    chunk_overlap_entry = tk.Entry(root, textvariable=chunk_overlap, bd=2, font=larger_font)
    chunk_overlap_entry.grid(row=4, column=1, sticky='w')

    use_clustering = tk.BooleanVar(value=False)
    def on_clustering_checkbox_click():
        current_state = use_clustering.get()
        logging.info(f"Clustering checkbox clicked. New state: {current_state}")
        
    clustering_check = ttk.Checkbutton(root, text="Use Clustering", variable=use_clustering,
                                       onvalue=True, offvalue=False,
                                       command=on_clustering_checkbox_click)
    clustering_check.grid(row=5, column=0, columnspan=2, pady=10, padx=10, sticky='w')

    # Function for file selection
    def file_select():
        global selected_file_path
        selected_file_path = select_file()
        if selected_file_path:
            api_key = get_api_key()
            if api_key:
                text = file_handler.load_document(selected_file_path)
                if text:
                    language = language_processing.detect_language(text)
                    custom_prompt = default_prompt_en
                    if language == "nl":
                        dutch_prompt = language_processing.translate_prompt(default_prompt_en, "nl")
                        custom_prompt = dutch_prompt if dutch_prompt else default_prompt_en

                    custom_prompt_area.delete("1.0", tk.END)
                    custom_prompt_area.insert(tk.END, custom_prompt)
                    progress['value'] = 0
                    summarize_button['state'] = 'normal'
                else:
                    messagebox.showerror("Error", "Failed to load document.")
                    summarize_button['state'] = 'disabled'
            else:
                messagebox.showerror("Error", "API Key Missing or Invalid.")
                summarize_button['state'] = 'disabled'
        else:
            summarize_button['state'] = 'disabled'

    select_button = ttk.Button(root, text="Select Document", command=file_select, style='W.TButton')
    select_button.grid(row=6, column=0, columnspan=2, pady=20, padx=10, sticky='ew')

    summarize_button = ttk.Button(root, text="Start Summarization", command=lambda: start_summarization_thread(root), style='W.TButton')
    summarize_button.grid(row=7, column=0, columnspan=2, pady=20, padx=10, sticky='ew')

    root.mainloop()

# Script Execution Block
# This block is the entry point of the application. It initializes the GUI and starts the application.
if __name__ == '__main__':
    logging.info("Starting the Document Summarizer application.")
    main_gui()
