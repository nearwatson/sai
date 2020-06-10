from flask import Flask, request

#实例化flask对象
app = Flask(__name__)



# status_model = 















@app.route('/indexabc', methods=['GET', 'POST'])
def index():
    inputs = request.form['text']
    # status = status_model(inputs)
    # return {'key': request.form['text']}
    return {'key': request.form['text']}















if __name__ == '__main__':
    app.run(debug=True)