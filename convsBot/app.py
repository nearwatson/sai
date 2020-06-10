from flask import Flask, request

#实例化flask对象
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    return {'key': request.form['text']}

if __name__ == '__main__':
    app.run(debug=True)