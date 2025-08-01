import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import boundingboxes

class SingleItemDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Single Image AI Detection")

        tk.Label(root, text="Select an image to analyze").pack(pady=10)

        self.choose_btn = tk.Button(root, text="Choose Image", command=self.choose_image)
        self.choose_btn.pack()

        self.canvas = tk.Canvas(root, width=640, height=640, bg="white")
        self.canvas.pack()

        self.result_label = tk.Label(root, text="", wraplength=600, justify="left")
        self.result_label.pack(pady=10)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        try:
            detections, output_img, ai_results = boundingboxes.run_detection(file_path)

            # Show image
            img = Image.open(output_img).resize((640, 640))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            # Show results
            result_lines = []
            for i, det in enumerate(detections):
                line = f"[Crop {i}] Class {det['class_id']} - Score: {det['score']:.2f} → "
                line += "AI" if det["ai_like"] else "Human"
                line += f" (AI Score: {det['ai_score']:.2f})"
                result_lines.append(line)

            self.result_label.config(text="\n".join(result_lines))

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = SingleItemDetectionApp(root)
    root.mainloop()
