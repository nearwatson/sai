from flask import Flask, request
from flask import jsonify
import json

import os
#实例化flask对象
app = Flask(__name__)

# status_model = 

@app.route('/', methods=['GET', 'POST'])
def index():
    inputs = request.get_json()

    print(json.dumps(inputs))
    inputs = json.dumps(inputs)
    # print(type(inputs), inputs, jsonify(inputs), type(jsonify(inputs)))
    # status = status_model(inputs)
    # return {'key': request.form['text']}
    return inputs # jsonify(inputs)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', 5000)))
