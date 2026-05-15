import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# CNN Model
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5)

# GUI App
class DigitRecognizer:

    def __init__(self, root):

        self.root = root
        self.root.title("Handwritten Digit Recognition")

        self.canvas_width = 300
        self.canvas_height = 300

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg='black'
        )

        self.canvas.grid(row=0, column=0)

        self.label = tk.Label(
            root,
            text="Draw Digit",
            font=("Arial", 20)
        )

        self.label.grid(row=0, column=1, padx=20)

        predict_btn = tk.Button(
            root,
            text="Predict",
            command=self.predict_digit
        )

        predict_btn.grid(row=1, column=1)

        clear_btn = tk.Button(
            root,
            text="Clear",
            command=self.clear_canvas
        )

        clear_btn.grid(row=2, column=1)

        # PIL image for accurate drawing
        self.image = Image.new(
            "L",
            (self.canvas_width, self.canvas_height),
            color=0
        )

        self.draw_image = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def draw_lines(self, event):

        x = event.x
        y = event.y

        r = 10

        self.canvas.create_oval(
            x-r,
            y-r,
            x+r,
            y+r,
            fill='white',
            outline='white'
        )

        self.draw_image.ellipse(
            [x-r, y-r, x+r, y+r],
            fill=255
        )

    def clear_canvas(self):

        self.canvas.delete("all")

        self.image = Image.new(
            "L",
            (self.canvas_width, self.canvas_height),
            color=0
        )

        self.draw_image = ImageDraw.Draw(self.image)

        self.label.config(text="Draw Digit")

    def predict_digit(self):

        # Resize to MNIST size
        img = self.image.resize((28, 28))

        # Convert to numpy
        img = np.array(img)

        # Normalize
        img = img / 255.0

        # Reshape
        img = img.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img)

        digit = np.argmax(prediction)

        confidence = np.max(prediction)

        self.label.config(
            text=f"Digit: {digit}\nConfidence: {confidence:.2f}"
        )

# Run App
root = tk.Tk()

app = DigitRecognizer(root)

root.mainloop()