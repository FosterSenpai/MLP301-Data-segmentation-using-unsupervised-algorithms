import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

class GUI:
    def __init__(self, root_window):
        # Window setup
        self.root = root_window
        self.root.title("Plant Leaf Image Clustering")
        self.root.geometry('950x500')
        self.root.resizable(False, False)
        self.root.configure(bg='#f0f0f0')
        # == Frames ==
        # Left Frame: Buttons and dropdown
        left_frame = tk.Frame(self.root, width=200, bg="#d9d9d9")
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        left_frame.pack_propagate(False)
        # Right Frame: Image display
        right_frame = tk.Frame(self.root, width=700, bg="#d9d9d9")
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        # == Image Display ==
        # Image display frames: grid, original on top, 3 images below
        image_display_frame = tk.Frame(right_frame, bg=right_frame['bg'])
        image_display_frame.pack(pady=10)
        self.image_labels = {}
        # Top: Original (spans 3 cols)
        frame_orig = tk.LabelFrame(image_display_frame, text="Original", width=200, height=200, bg=right_frame['bg'])
        frame_orig.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        frame_orig.pack_propagate(False) # Need to stop resizing
        frame_orig.grid_propagate(False)
        label_orig = tk.Label(frame_orig, bg="white", width=180, height=180)
        label_orig.pack(fill="both", expand=True)
        self.image_labels["Original"] = label_orig
        # Bottom: Entire, Healthy, Diseased
        for i, (title, key) in enumerate([("Entire", "Entire"), ("Healthy", "Healthy"), ("Diseased", "Diseased")]):
            frame = tk.LabelFrame(image_display_frame, text=title, width=200, height=200, bg=right_frame['bg'])
            frame.grid(row=1, column=i, padx=5, pady=5)
            frame.pack_propagate(False)
            frame.grid_propagate(False)
            label = tk.Label(frame, bg="white", width=180, height=180)
            label.pack(fill="both", expand=True)
            self.image_labels[key] = label
        # == Controls ==
        # Select folder button
        tk.Button(left_frame, text="Select Folder", command=self.select_folder).pack(pady=10, fill="x", padx=10)
        dropdown_frame = tk.Frame(left_frame, bg="#d9d9d9")
        dropdown_frame.pack(pady=10, fill="x", padx=10)
        # Dropdown menu for image selection
        tk.Label(dropdown_frame, text="Select Image:", bg="#d9d9d9").pack(anchor="w")
        self.dropdown_var = tk.StringVar(self.root)
        self.dropdown_menu = ttk.Combobox(dropdown_frame, textvariable=self.dropdown_var, state="readonly", width=20)
        self.dropdown_menu.pack(fill="x")
        self.dropdown_menu.bind("<<ComboboxSelected>>", self.on_dropdown_change)
        # Elbow method button
        tk.Button(left_frame, text="Show Elbow Method", command=self.elbow_method).pack(pady=10, fill="x", padx=10)
        tk.Button(left_frame, text="Exit", command=self.root.destroy).pack(pady=40, fill="x", padx=10)
        
        # == Data ==
        self.image_paths = [] # List of image file paths
        self.current_index = 0

    def select_folder(self):
        # Prompt for directory
        dir_path = filedialog.askdirectory()
        if not dir_path: # User cancelled
            return
        
        # Get image files in directory
        self.image_paths = [] # Reset list
        filenames = []
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                full_path = os.path.join(dir_path, filename)
                self.image_paths.append(full_path)
                filenames.append(filename)
        
        # Update dropdown menu
        if filenames: # Images found, set dropdown
            self.dropdown_menu['values'] = filenames # Set dropdown values
            self.dropdown_menu.current(0)            # Select first by default
            self.current_index = 0
            self.display_image_set()                 # Display first image set
        else: # No images found, show warning
            messagebox.showwarning("No Images", "No .jpg or .jpeg images found in the selected folder.")

    def on_dropdown_change(self, event=None):
        # Get selected filename from dropdown
        selected_filename = self.dropdown_var.get()
        # Find index of selected filename
        for idx, path in enumerate(self.image_paths):
            if os.path.basename(path) == selected_filename:
                self.current_index = idx
                break
        # Display set for selected image
        self.display_image_set()

    def display_image_set(self):
        # Check if images are available
        if not self.image_paths:
            return

        # Load and display original image
        image_path = self.image_paths[self.current_index]
        pil_image = Image.open(image_path)
        self.display_image(pil_image, "Original")

        pre_img = self.preprocess_image(pil_image)
        seg_img, seg_labels = self.segment_image(pre_img, k=3)
        leaf_mask, healthy_mask, diseased_mask = self.postprocess_image(pre_img, seg_labels)

        # Convert masks to PIL Images (mode 'L' for grayscale)
        entire_pil = Image.fromarray(leaf_mask).convert("L")
        healthy_pil = Image.fromarray(healthy_mask).convert("L")
        diseased_pil = Image.fromarray(diseased_mask).convert("L")

        self.display_image(entire_pil, "Entire")
        self.display_image(healthy_pil, "Healthy")
        self.display_image(diseased_pil, "Diseased")

    def display_image(self, image, label_key):
        # Resize to fit label
        image = image.resize((180, 180), Image.Resampling.LANCZOS)
        # Convert to PhotoImage
        photo_image = ImageTk.PhotoImage(image)
        
        # Set image in label
        label = self.image_labels[label_key]
        label.config(image=photo_image)
        label.image = photo_image

    def preprocess_image(self, img):
       # Resize
        img = img.resize((256, 256))
        img = np.array(img)

        # Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Histogram Equalization
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(gray)
        
        # Use morphological operations to rmeove shadows
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        img = cv2.subtract(background, gray)

        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        
        # Convert back to 3-channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Resize
        # Color space conversion
        # contrast enhancement
        # color correction
        # shadow removal
        # sharpening
        return img
    
    def get_leaf_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # Threshold on saturation (leaves are usually more saturated)
        _, s_mask = cv2.threshold(hsv[...,1], 40, 255, cv2.THRESH_BINARY)
        # Optional: threshold on hue for green/yellow leaves
        # _, h_mask = cv2.threshold(hsv[...,0], 25, 255, cv2.THRESH_BINARY)
        mask = s_mask
        # Fill holes and remove small objects
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Remove small objects (using skimage)
        from skimage import morphology
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=2000)
        mask = (mask * 255).astype(np.uint8)
        return mask

    def segment_image(self, img, k=3):
        leaf_mask = self.get_leaf_mask(img)
        mask_bool = leaf_mask > 0
        pixels = img[mask_bool].reshape(-1, 3).astype(np.float32)
        if len(pixels) == 0:
            # fallback: cluster whole image
            pixels = img.reshape(-1, 3).astype(np.float32)
            mask_bool = np.ones(img.shape[:2], dtype=bool)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) # type: ignore
        centers = np.uint8(centers)
        segmented = np.zeros_like(img)
        segmented_pixels = centers[labels.flatten()] # type: ignore
        segmented[mask_bool] = segmented_pixels
        # Create a full-size label mask, -1 for background
        full_labels = np.full(img.shape[:2], -1, dtype=np.int32)
        full_labels[mask_bool] = labels.flatten()
        return segmented, full_labels

    def postprocess_image(self, pre_img, seg_labels):
        leaf_mask = self.get_leaf_mask(pre_img)
        if np.mean(leaf_mask) < 127:
            leaf_mask = cv2.bitwise_not(leaf_mask)
        unique, counts = np.unique(seg_labels, return_counts=True)
        healthy_cluster = unique[np.argmax(counts)]
        healthy_mask = ((seg_labels == healthy_cluster) & (leaf_mask > 0)).astype(np.uint8)
        diseased_mask = ((seg_labels != healthy_cluster) & (seg_labels >= 0) & (leaf_mask > 0)).astype(np.uint8)
        # Morphological closing and remove small objects
        kernel = np.ones((7,7), np.uint8)
        healthy_mask = cv2.morphologyEx(healthy_mask, cv2.MORPH_CLOSE, kernel)
        diseased_mask = cv2.morphologyEx(diseased_mask, cv2.MORPH_CLOSE, kernel)
        from skimage import morphology
        healthy_mask = morphology.remove_small_objects(healthy_mask.astype(bool), min_size=500)
        diseased_mask = morphology.remove_small_objects(diseased_mask.astype(bool), min_size=200)
        healthy_mask = (healthy_mask * 255).astype(np.uint8)
        diseased_mask = (diseased_mask * 255).astype(np.uint8)
        return leaf_mask, healthy_mask, diseased_mask
        

    def elbow_method(self):
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please select a folder and image first.")
            return

        image_path = self.image_paths[self.current_index]
        pil_image = Image.open(image_path)
        pre_img = self.preprocess_image(pil_image)
        pixels = pre_img.reshape(-1, 3).astype(np.float32)

        k_values = list(range(2, 11))
        compactness_values = []
        segmented_images = []

        for k in k_values:
            compactness, labels, centers = cv2.kmeans(
                pixels, k, None, # type: ignore
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                10, cv2.KMEANS_RANDOM_CENTERS
            ) # type: ignore
            compactness_values.append(compactness)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()] # type: ignore
            segmented_img = segmented_data.reshape(pre_img.shape)
            segmented_images.append(segmented_img)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 6))

        # Plot inertia curve in the first subplot
        plt.subplot(2, 5, 1)
        plt.plot(k_values, compactness_values, marker='o')
        plt.title("Elbow Method")
        plt.xlabel("k")
        plt.ylabel("Compactness")
        plt.grid(True)

        # Show segmented images for each k in the remaining 9 subplots
        for i, (k, seg_img) in enumerate(zip(k_values, segmented_images)):
            plt.subplot(2, 5, i+2)
            plt.imshow(seg_img)
            plt.title(f"k={k}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
    