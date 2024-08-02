import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('model.keras')
expressions = ['happy','sad','angry','fear','surprised','neutral','disgust']

def face(img):
    if img is not None:
        # Convert PIL image to the format expected by the model
        img = img.convert('L')  # Ensure image is in grayscale
        img = img.resize((48, 48))  # Resize to (48, 48)
        img = np.array(img)  # Convert image to array
        img = img.reshape((1, 48, 48, 1)).astype('float32') / 255
        prediction = model.predict(img)
        return {str(i): float(prediction[0][i]) for i in range(7)}
    else:
        return ''

# Create the Gradio interface
iface = gr.Interface(
    fn=face,
    inputs=gr.Image(type='pil'),  # Use type='pil' to handle PIL images
    outputs=gr.Label(num_top_classes=7,label=expressions),
    live=True 
)

# Launch the interface
iface.launch()
