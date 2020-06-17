from flask import Flask, request
from flask import jsonify
import json
import os
import model_predict
from model_predict import TEXT, predict_model, data2sent

from utils import *
import torch
#实例化flask对象
app = Flask(__name__)

# status_model = 



@app.route('/', methods=['GET', 'POST'])
def index():
#     print(dirrs(request))
#     data = TEXT.numericalize(inputs['text'])
    result = {"text":"hello"}
#     result = {"text":data2sent(predict_model(data), func=lambda word_tensor: torch.argmax(word_tensor, dim=-1))}
    # print(json.dumps(inputs))
    result = json.dumps(result)
    # print(type(inputs), inputs, jsonify(inputs), type(jsonify(inputs)))
    # status = status_model(inputs)
    # return {'key': request.form['text']}
    return result  # inputs

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', 5000)))
#     app.run(port=int(os.getenv('PORT', 5000)))
