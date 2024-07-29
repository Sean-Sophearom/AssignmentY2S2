from flask import Flask, render_template, request, jsonify
from model.model import main
import webbrowser

app = Flask(__name__, template_folder='./web', static_folder='./web')

user_count = 0

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/main")
def main_page():
    return render_template("main.html")

@app.route('/api', methods=['POST'])
def api_route():
    if request.headers['Content-Type'] == 'application/json':
        data = request.get_json()

        print(data)
        
        #get user id from cookie
        user_id = request.cookies.get('user_id')
        if user_id is None:
            global user_count
            user_count += 1
            user_id = str(user_count)

        response = main(user_id, data)        
        resp = jsonify(response)
        resp.set_cookie('user_id', user_id)

        return resp, 200
    else:
        return 'Content-Type not supported', 400

if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(host="localhost", port=5000, debug=False)  # Start the Flask app