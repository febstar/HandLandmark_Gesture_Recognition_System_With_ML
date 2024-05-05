import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import subprocess
from threading import Thread

script_mapping = {
    "Image Collection & Labelling": "image_collection.py",
    "Configure & Training": "configure_set.py",
    "Live Detection": "live_dec.py"
}

def run_python_script(script_name, output_widget):
    """ Function to run a python script and display output in real-time """
    try:
        process = subprocess.Popen(['python', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Function to read output from the process
        def stream_output():
            for line in iter(process.stdout.readline, ''):
                output_widget.config(state='normal')
                output_widget.insert(tk.END, line)
                output_widget.see(tk.END)  # Scroll to the end of the output
                output_widget.config(state='disabled')
            # Catch any error output after the regular output
            errors = process.stderr.read()
            if errors:
                output_widget.config(state='normal')
                output_widget.insert(tk.END, "\nErrors:\n" + errors)
                output_widget.see(tk.END)
                output_widget.config(state='disabled')
            process.stdout.close()
            process.stderr.close()

        # Start the thread to read output
        thread = Thread(target=stream_output)
        thread.start()

    except Exception as e:
        output_widget.config(state='normal')
        output_widget.insert(tk.END, f"Failed to run script: {str(e)}")
        output_widget.config(state='disabled')

def on_button_click(button_name, output_widget):
    global intro_frame, main_frame
    print(f"{button_name} button clicked!")
    if button_name == "Start":
        intro_frame.pack_forget()
        main_frame.pack(fill=tk.BOTH, expand=True)
    elif button_name in script_mapping:
        output_widget.config(state='normal')
        output_widget.delete('1.0', tk.END)  # Clear previous output
        output_widget.config(state='disabled')
        run_python_script(script_mapping[button_name], output_widget)
    else:
        output_widget.config(state='normal')
        output_widget.insert(tk.END, f"No script defined for {button_name}")
        output_widget.config(state='disabled')

def setup_ui():
    global intro_frame, main_frame, button_mapping

    root = tk.Tk()
    root.title("Gesture Recognition System")
    root.geometry("600x600")
    root.resizable(True, True)

    intro_frame = ttk.Frame(root)
    intro_label = ttk.Label(intro_frame, text="Welcome to the Gesture Recognition System!\nClick Start to proceed.", font=('Arial', 16))
    intro_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    start_button = ttk.Button(intro_frame, text="Start", command=lambda: on_button_click("Start", output))
    start_button.pack(side=tk.BOTTOM, pady=20)
    intro_frame.pack(fill=tk.BOTH, expand=True)

    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    # Output text area
    output = scrolledtext.ScrolledText(main_frame, height=10, state='disabled')
    output.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

    try:
        original_image = Image.open("Dataset/images/logo.png")
        resized_image = original_image.resize((150, 150), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(resized_image)
        logo_label = ttk.Label(main_frame, image=logo)
        logo_label.image = logo
        logo_label.grid(row=1, column=0, pady=20)
    except Exception as e:
        print(f"Error loading logo image: {e}")
        logo_label = ttk.Label(main_frame, text="Logo Image Missing")
        logo_label.grid(row=1, column=0, pady=20)

    buttons = {}
    for i, (button_text, _) in enumerate(script_mapping.items(), start=2):
        button = ttk.Button(main_frame, text=button_text, command=lambda bt=button_text: on_button_click(bt, output), style='TButton')
        button.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=8, padx=10)
        buttons[button_text] = button

    button_mapping = buttons

    root.mainloop()

if __name__ == "__main__":
    setup_ui()
