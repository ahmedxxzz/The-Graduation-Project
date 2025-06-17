import customtkinter as ctk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras.applications import vgg19, resnet50, vgg16, mobilenet_v2, xception, efficientnet, densenet
import threading 



def predict_image(model_path, image_path, progress_callback=None, total_steps=1, current_step=1):
    """
    Predict the class of a single given image using a given model.

    Args:
        model_path (str): Path to the Keras model file.
        image_path (str): Path to the image file.
        progress_callback (callable, optional): A callback function to show progress.
            The function should take 3 arguments: message (str), current step (int), total steps (int),
            and an optional is_error (bool).
        total_steps (int, optional): Total number of steps. Defaults to 1.
        current_step (int, optional): Current step. Defaults to 1.

    Returns:
        list: A list containing the model name, image name, and predicted class.
            like:
                    ['VGG19', 'o1.png', 'overriding']
    """
    
    try:
        model_name = os.path.basename(model_path).split('.')[0]

        img_h, img_w = (299, 299) if model_name == 'Xception' else (224, 224)

        model_config = {
            'VGG19': vgg19.preprocess_input,
            'ResNet50': resnet50.preprocess_input,
            'VGG16': vgg16.preprocess_input,
            'MobileNetV2': mobilenet_v2.preprocess_input,
            'Xception': xception.preprocess_input,
            'EfficientNetB0': efficientnet.preprocess_input,
            'DenseNet121': densenet.preprocess_input
        }

        if model_name not in model_config:
            raise ValueError(f"Preprocessing function not defined for model: {model_name}")
        preprocess_input_func = model_config[model_name]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = models.load_model(model_path)

        img = image.load_img(image_path, target_size=(img_h, img_w))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input_func(img_array)

        prediction = model.predict(img_preprocessed, verbose=0)
        
        img_name = os.path.basename(image_path)
        result_label = 'overriding' if prediction[0][0] > 0.5 else 'oblique'

        if progress_callback:
            progress_callback(f"Predicted: {img_name} with {model_name} -> {result_label}", current_step, total_steps)

        return [model_name, img_name, result_label]

    except Exception as e:
        error_msg = f"Error predicting {os.path.basename(image_path)} with {model_name}: {e}"
        if progress_callback:
            progress_callback(error_msg, current_step, total_steps, is_error=True)
        return [model_name, os.path.basename(image_path), f"Error: {e}"]


def handle_pipes_gui(model_name_selected, path_selected, progress_callback=None):
    """
    Run predictions on a single image or a directory of images using one or all of the available models.

    Args:
        model_name_selected (str): The name of the model to use for predictions. If 'All Models', all available models will be used.
        path_selected (str): The path to the image or directory containing images to be processed.
        progress_callback (callable, optional): A callback function to show progress. The function should take 3 arguments: message (str), current step (int), total steps (int), and an optional is_error (bool).

    Returns:
        list: A list of predictions, where each prediction is a list containing the model name, image name, and predicted class label ('overriding' or 'oblique').

        predictions = [
                ['VGG19', 'image.jpg', 'overriding'],
                ['ResNet50', 'image.jpg', 'oblique'],
                ]
                باختصار بيحدد انهى موديل هيستخدم ويشوف الصورة او الصور اللى هتستخدم  , وبعد بيعمل لووب ويلف على الصور ويباصى لل معادلة اللى بتتوقعلى 
                وبعد كدا يجمع النتائج فى الليست
    """
    
    AVAILABLE_MODELS = ["DenseNet121", "EfficientNetB0", "MobileNetV2",
                        "ResNet50", "VGG16", "VGG19", "Xception"]
    predictions = []
    
    # Determine if path_selected is a single image or a directory
    is_single_image = os.path.isfile(path_selected)
    
    models_to_run = []
    if model_name_selected == "All Models":
        models_to_run = AVAILABLE_MODELS
    else:
        models_to_run = [model_name_selected]

    # --- Determine images to process ---
    images_to_process = []
    if is_single_image:
        images_to_process = [path_selected]
    else:
        try:
            image_files = [os.path.join(path_selected, f) for f in os.listdir(path_selected) 
                           if os.path.isfile(os.path.join(path_selected, f)) and 
                           any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'])]
            if not image_files:
                if progress_callback:
                    progress_callback("No image files found in the selected directory.", 0, 1, is_error=True)
                return []
            images_to_process = image_files
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error reading directory {path_selected}: {e}", 0, 1, is_error=True)
            return []

    total_predictions_to_make = len(models_to_run) * len(images_to_process)
    current_prediction_count = 0


    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_base_dir = os.path.join(script_dir, "Saved Models")

    if not os.path.isdir(models_base_dir):
        if progress_callback:
            progress_callback(f"Error: 'Saved Models' directory not found at {models_base_dir}", 0, 1, is_error=True)
        return []

    for model_n in models_to_run:
        model_file_path = os.path.join(models_base_dir, f"{model_n}.h5")
        if not os.path.exists(model_file_path):
            msg = f"Model file {model_n}.h5 not found in '{models_base_dir}'. Skipping."
            if progress_callback:
                for _ in images_to_process:
                    current_prediction_count += 1
                    progress_callback(msg, current_prediction_count, total_predictions_to_make, is_error=True)
            predictions.extend([[model_n, os.path.basename(img_p), "Model file missing"] for img_p in images_to_process])
            continue 

        for img_p in images_to_process:
            current_prediction_count += 1
            if progress_callback:
                progress_callback(f"Processing: {os.path.basename(img_p)} with {model_n}...", current_prediction_count, total_predictions_to_make)
            
            prediction_result = predict_image(model_file_path, img_p, progress_callback, total_predictions_to_make, current_prediction_count)
            predictions.append(prediction_result)
            
    if not predictions and progress_callback: 
         progress_callback("No predictions were made. Check logs or input.", 0, 1, is_error=True)

    return predictions

# --- GUI Application ---
class ImagePredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Predictor")
        self.geometry("800x700")

        ctk.set_appearance_mode("Dark")  
        ctk.set_default_color_theme("blue")

        self.selected_path = ctk.StringVar()
        self.selected_model = ctk.StringVar(value="All Models") 

        # --- Main frame ---
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # --- Top controls frame ---
        controls_frame = ctk.CTkFrame(main_frame)
        
        controls_frame.pack(pady=10, padx=10, fill="x")

        # Model Selection
        ctk.CTkLabel(controls_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        model_options = ["All Models", "DenseNet121", "EfficientNetB0", "MobileNetV2",
                         "ResNet50", "VGG16", "VGG19", "Xception"]
        self.model_menu = ctk.CTkComboBox(controls_frame, variable=self.selected_model, values=model_options, width=200)
        self.model_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Path Selection
        ctk.CTkLabel(controls_frame, text="Image/Folder Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.path_label = ctk.CTkEntry(controls_frame, textvariable=self.selected_path, state="readonly", width=300)
        self.path_label.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = ctk.CTkButton(controls_frame, text="Browse", command=self.browse_path)
        self.browse_button.grid(row=1, column=2, padx=5, pady=5)

        controls_frame.grid_columnconfigure(1, weight=1)

        # --- Image Preview (Optional) ---
        self.image_preview_label = ctk.CTkLabel(main_frame, text="Image Preview (select a single image file)", height=200, width=200)
        self.image_preview_label.pack(pady=10)
        self.ctk_image = None 

        # --- Predict Button ---
        self.predict_button = ctk.CTkButton(main_frame, text="Start Prediction", command=self.start_prediction_thread, state="disabled")
        self.predict_button.pack(pady=10)

        # --- Progress Bar and Status ---
        self.progress_bar = ctk.CTkProgressBar(main_frame, orientation="horizontal", mode="determinate")
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=5, padx=10, fill="x")
        
        self.status_label = ctk.CTkLabel(main_frame, text="Status: Idle")
        self.status_label.pack(pady=5, padx=10, fill="x")


        # --- Results Treeview ---
        results_frame = ctk.CTkFrame(main_frame)
        results_frame.pack(pady=10, padx=10, fill="both", expand=True)

        ctk.CTkLabel(results_frame, text="Prediction Results:").pack(anchor="w", padx=5)

        style = ttk.Style()
        fg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        text_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        header_bg = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        
        style.theme_use("default") 
        style.configure("Treeview",
                        background=fg_color,
                        foreground=text_color,
                        fieldbackground=fg_color,
                        rowheight=25)
        style.configure("Treeview.Heading",
                        background=header_bg,
                        foreground=text_color,
                        relief="flat")
        style.map("Treeview.Heading", background=[('active', self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["hover_color"]))])


        self.tree = ttk.Treeview(results_frame, columns=("Model", "Image", "Prediction"), show="headings")
        self.tree.heading("Model", text="Model")
        self.tree.heading("Image", text="Image File")
        self.tree.heading("Prediction", text="Prediction")

        self.tree.column("Model", width=150, anchor="w")
        self.tree.column("Image", width=250, anchor="w")
        self.tree.column("Prediction", width=150, anchor="w")

        vsb = ctk.CTkScrollbar(results_frame, command=self.tree.yview)
        hsb = ctk.CTkScrollbar(results_frame, command=self.tree.xview, orientation="horizontal")
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)
        
        self.prediction_thread = None


    def browse_path(self):
        # Ask user if they want to select a file or folder
        choice = messagebox.askquestion("Select Path Type", "Select a single image file?\n(Choose 'No' to select a folder of images)", icon='question')
        
        path = ""
        if choice == 'yes': # Select single file
            path = filedialog.askopenfilename(
                title="Select an Image File",
                filetypes=(("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff"), ("All files", "*.*"))
            )
            if path:
                self.display_image_preview(path)
            else:
                self.clear_image_preview()
        else: # Select folder
            path = filedialog.askdirectory(title="Select a Folder Containing Images")
            self.clear_image_preview()

        if path:
            self.selected_path.set(path)
            self.predict_button.configure(state="normal")
        else:
          
            if not self.selected_path.get():
                self.predict_button.configure(state="disabled")

    def display_image_preview(self, image_path):
        try:
            img = Image.open(image_path)
            
            # Resize for preview
            max_width = 200
            max_height = 200
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            self.ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            self.image_preview_label.configure(image=self.ctk_image, text="")
        except Exception as e:
            self.image_preview_label.configure(image=None, text=f"Cannot preview:\n{os.path.basename(image_path)}\nError: {e}")
            self.ctk_image = None 

    def clear_image_preview(self):
        self.image_preview_label.configure(image=None, text="Image Preview (select a single image file)")
        self.ctk_image = None

    def update_progress(self, message, current_step, total_steps, is_error=False):
        self.status_label.configure(text=message, text_color="red" if is_error else "white")
        if total_steps > 0:
            progress_value = current_step / total_steps
            self.progress_bar.set(progress_value)
        else:
            self.progress_bar.set(0) 
        self.update_idletasks() 

    def start_prediction_thread(self):
        if not self.selected_path.get():
            messagebox.showerror("Error", "Please select an image or folder path.")
            return
        if not self.selected_model.get():
            messagebox.showerror("Error", "Please select a model.")
            return

        
        self.predict_button.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.model_menu.configure(state="disabled")

        
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.progress_bar.set(0)
        self.status_label.configure(text="Status: Starting predictions...")

        
        self.prediction_thread = threading.Thread(target=self.run_predictions, daemon=True)
        self.prediction_thread.start()
        

    def run_predictions(self):
        model_to_use = self.selected_model.get()
        path_to_process = self.selected_path.get()
        
        try:
            results = handle_pipes_gui(model_to_use, path_to_process, self.update_progress_gui_safe)
            
            self.after(0, self.finalize_predictions, results)

        except Exception as e:
            
            self.after(0, self.update_progress_gui_safe, f"Critical error during prediction: {e}", 0, 1, True)
            self.after(0, self.enable_controls) 

    def update_progress_gui_safe(self, message, current_step, total_steps, is_error=False):

        self.after(0, self.update_progress, message, current_step, total_steps, is_error)

    def finalize_predictions(self, results):
        if not results:
            self.status_label.configure(text="Status: No results or an error occurred. Check messages.")
        else:
            for res in results:
                self.tree.insert("", "end", values=res)
            self.status_label.configure(text=f"Status: Predictions complete. {len(results)} results found.")
            self.progress_bar.set(1) 

        self.enable_controls()

    def enable_controls(self):
        self.predict_button.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.model_menu.configure(state="normal")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    saved_models_dir = os.path.join(script_dir, "Saved Models")
    app = ImagePredictorApp()
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)
        print(f"Created 'Saved Models' directory at: {saved_models_dir}")
        print("Please place your .h5 model files (e.g., VGG19.h5) in this directory.")
    
    dummy_models = ["DenseNet121.h5", "EfficientNetB0.h5", "MobileNetV2.h5","ResNet50.h5", "VGG16.h5", "VGG19.h5", "Xception.h5"]

    app.mainloop()



