import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from papzi.constants import BASE_DIR
from papzi_gui.face_recognizer import FaceRecognizer

# Ensure the Roboto font is installed on your system
FONT_NAME = "Roboto"

POSTER_SAMPLE_PATH = r"/Users/eisenheixm/own/music-recommendation-engine/src/papzi/data/posters/Alexandra Daddario/tv/Mayfair Witches_2023_7.4.jpg"


class PhotoSelectorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Photo Selector and Calculator")
        self.geometry("400x600")
        self.configure(bg="#2E2E2E")

        # Define styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "TButton",
            font=(FONT_NAME, 12),
            background="#555555",
            foreground="white",
            borderwidth=1,
            focusthickness=3,
            focuscolor="none",
            padding=10,
        )
        self.style.map("TButton", background=[("active", "#444444")])

        self.create_widgets()
        self.create_buttons()
        self.recognizer = FaceRecognizer()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, bg="#2E2E2E")
        self.scrollable_frame = tk.Frame(self.canvas, bg="#2E2E2E")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )

        self.canvas_frame = self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw",
            width=self.canvas.winfo_reqwidth(),  # Ensure the window fills the canvas initially
        )

        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)

        # After packing, update the width of the window to match the canvas
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_frame, width=e.width),
        )

    def update_scroll_region(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def create_buttons(self):
        # Create and pack the button to select photo
        self.select_button = ttk.Button(
            self.scrollable_frame,
            text="Select Photo",
            command=self.select_photo,
            style="TButton",
        )
        self.select_button.pack(pady=20, ipadx=40)  # Button width adjustment

        # Create and pack the image label (initially empty)
        self.image_label = tk.Label(self.scrollable_frame, bg="#2E2E2E")
        self.image_label.pack(pady=20)

        # Label to display the main result name
        self.actor_name = tk.Label(
            self.scrollable_frame,
            text="",
            font=(FONT_NAME, 14),
            bg="#2E2E2E",
            fg="white",
        )
        self.actor_name.pack(pady=10)

        # Frame to contain the list of movie names and posters
        self.film_frame = tk.Frame(self.scrollable_frame, bg="#2E2E2E")
        self.film_frame.pack(pady=10)
        # self.film_frame.bind("<Configure>", self.update_scroll_region)

    def select_photo(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Allowed Types", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not file_path:
            return
        file_path = Path(file_path)

        # Open the selected image
        img = Image.open(file_path)

        # Resize the image to fit within 200x200 pixels while maintaining aspect ratio
        img.thumbnail((200, 200), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        # Update the label to display the image
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

        # Calculations and results
        name = self.recognizer.predict(file_path)

        # Display the results
        self.actor_name.config(text=name)

        # Clear previous movie posters
        for widget in self.film_frame.winfo_children():
            widget.destroy()
        self.show_films(name)

    def get_films(self, name: str):
        movie_dir = BASE_DIR / "posters" / name / "movie"
        tv_dir = BASE_DIR / "posters" / name / "tv"
        if movie_dir.exists():
            movie_posters = list(movie_dir.iterdir())
        else:
            movie_posters = []
        if tv_dir.exists():
            tv_posters = list(tv_dir.iterdir())
        else:
            tv_posters = []
        all_posters = movie_posters + tv_posters
        all_films = []
        for poster in all_posters:
            name, year, rating = poster.name.replace(".jpg", "").split("_")
            all_films.append(
                {"name": name, "rating": rating, "poster": poster}
            )
            all_films.sort(key=lambda x: float(x["rating"]), reverse=True)
        return all_films[:10]

    def show_films(self, name: str):
        print(self.film_frame.winfo_height())
        # Display the results with movie posters
        all_films = self.get_films(name)
        # all_films.sort(key=lambda x: float(x["rating"]))
        for film in all_films:
            frame = tk.Frame(self.film_frame, bg="#2E2E2E")
            frame.pack(anchor="center", pady=5)

            movie_label = tk.Label(
                frame,
                text=film["name"],
                font=(FONT_NAME, 12),
                bg="#2E2E2E",
                fg="white",
            )
            movie_label.pack(side="left", padx=5)

            # Load and resize the poster sample image
            poster_img = Image.open(film["poster"])
            poster_img.thumbnail(
                (68, 100), Image.LANCZOS
            )  # Scaling height to 100 pixels
            poster_img_tk = ImageTk.PhotoImage(poster_img)
            movie_poster = tk.Label(frame, image=poster_img_tk)
            movie_poster.image = poster_img_tk
            movie_poster.pack(side="right", padx=5)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        print("films visible")

    def on_mouse_wheel(self, event):
        print("mouse wheel")
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def main():
    app = PhotoSelectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
