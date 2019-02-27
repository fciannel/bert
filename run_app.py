import grpc
import tensorflow as tf

import bert_squad_client as sq_client
from flask import Flask
from flask import request

import random

app = Flask(__name__)

def format_input_data(content):
    context = content['context']
    question = content['question']
    id = content['id']
    input_data = {"data": [{"title": "na", "paragraphs": [{"context": context, "qas": [{"answers": [], "id": id, "question": question }]}]}], "version": "1.1"}
    return input_data

@app.route("/", methods=['GET'])
def hello():
    return "Hello BERT predicting AG NEWS! Try posting a context/question to this url"

@app.route("/", methods=['POST'])
def predict():
    content = request.get_json()
    input_data = format_input_data(content)
    id = content['id']
    all_predictions, all_nbest_json = sq_client.run_single_prediction(input_data)
    print(all_predictions[id])
    return all_predictions[id]


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6006)


"""
This is how to run it:

curl -X POST \
  http://svl290-dd24-240.cisco.com:6007/ \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -H 'postman-token: 4e2181a9-9da8-12f2-eaf8-cd2382d253bc' \
  -d '{"context": "There would be no more scoring in the third quarter, but early in the fourth, the Broncos drove to the Panthers 41-yard line. On the next play, Ealy knocked the ball out of Manning'\''s hand as he was winding up for a pass, and then recovered it for Carolina on the 50-yard line. A 16-yard reception by Devin Funchess and a 12-yard run by Stewart then set up Gano'\''s 39-yard field goal, cutting the Panthers deficit to one score at 16\u201310. The next three drives of the game would end in punts.", "question": "Who recovered a Manning fumble?", "id":"1"}'
"""
