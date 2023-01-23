import torch
import requests
import io

#! wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
#! unzip -d weights -j weights.zip
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from flask import Flask, Response


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)

    # load and normalize image
    #input_image = Image.open('example_image.jpg')

    # or load from URL...
    image_url = 'https://farm5.staticflickr.com/4141/4856248695_03475782dc_z.jpg'
    input_image = Image.open(requests.get(image_url, stream=True).raw)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    img = transform(input_image).unsqueeze(0)

    prompts = ['a glass', 'something to fill', 'wood', 'a jar']

    # predict
    with torch.no_grad():
        preds = model(img.repeat(4,1,1,1), prompts)[0]

    #img_from_tensor = Image.fromarray(input_image, 'RGB')

    img_byte_arr = io.BytesIO()
    #img_from_tensor.save(img_byte_arr, format='PNG')
    input_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = Response(img_byte_arr)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
