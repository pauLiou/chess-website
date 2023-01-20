import base64
from model import main_processing


def binary_to_image_data(mime_type,b):
    base64_data = base64.encodebytes(b).decode('ascii')
    return f'data:{mime_type};base64,{base64_data}'

def run_chess_model(img):
    predicted_img = main_processing.run_classification(img)
    return predicted_img