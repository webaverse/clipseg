import torch
import requests
import io

#! wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
#! unzip -d weights -j weights.zip
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from flask import Flask, request, Response
import json

app = Flask(__name__)

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if (request.method == 'OPTIONS'):
        # print('got options 1')
        response = flask.Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        # print('got options 2')
        return response

    body = request.get_data()
    # load and normalize image
    input_image = Image.open(io.BytesIO(body))

    # parse prompts argument which should be encoded as json
    prompts_json_string = request.args.get('prompts')
    prompts = json.loads(prompts_json_string)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    img = transform(input_image).unsqueeze(0)

    # prompts = ['a glass', 'something to fill', 'wood', 'a jar']

    # predict
    preds = None
    with torch.no_grad():
        preds = model(img.repeat(4,1,1,1), prompts)[0]

    #img_from_tensor = Image.fromarray(input_image, 'RGB')

    print(f"preds.shape: {preds.shape}")

    img_byte_arr = io.BytesIO()
    #img_from_tensor.save(img_byte_arr, format='PNG')
    input_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = Response(img_byte_arr)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
