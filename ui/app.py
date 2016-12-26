from flask import Flask, render_template, request
from flask import jsonify

import os
import sys

from tf_seq2seq_chatbot.configs.config import FLAGS
from tf_seq2seq_chatbot.lib import data_utils
from tf_seq2seq_chatbot.lib.seq2seq_model_utils import create_model, get_predicted_sentence

#======================= Chatterbot files ============================

# -*- coding: utf-8 -*-
from chatterbot import ChatBot
import logging
from chatterbot.trainers import ChatterBotCorpusTrainer
#====================================================================================

#======================= Chatterbot code ============================

# Uncomment the following line to enable verbose logging
# logging.basicConfig(level=logging.INFO)

# Create a new instance of a ChatBot
bot = ChatBot("Terminal",
    storage_adapter="chatterbot.storage.JsonFileStorageAdapter",
    logic_adapters=[
        "chatterbot.logic.MathematicalEvaluation",
        #"chatterbot.logic.TimeLogicAdapter",
        "chatterbot.logic.BestMatch",

    ],
    input_adapter="chatterbot.input.TerminalAdapter",
    output_adapter="chatterbot.output.TerminalAdapter",
    database="../../database.db",
    read_only = True
)

bot.set_trainer(ChatterBotCorpusTrainer)
bot.train(
    "chatterbot.corpus.english.greetings",
    "chatterbot.corpus.english.conversations"
)

#======================================================================================

app = Flask(__name__,static_url_path="/static") 

#############
# Routing
#
@app.route('/message', methods=['POST'])
def reply():
    return jsonify( { 'text': predict(request.form['msg']) } )

@app.route("/")
def index(): 
    return render_template("index.html")
#############
def predict(txt):
    global vocab, rev_vocab, rev_vocab, model, sess
    predicted_sentence = get_predicted_sentence(txt, vocab, rev_vocab, model, sess)
    if "_UNK" in predicted_sentence:
        predicted_sentence = bot.get_response(txt)
    return predicted_sentence

'''
Init seq2seq model

    1. Call main from execute.py
    2. Create decode_line function that takes message as input
'''
#_________________________________________________________________
import tensorflow as tf
#import execute

sess = tf.Session()
# Create model and load parameters.
model = create_model(sess, forward_only=True)
model.batch_size = 1  # We decode one sentence at a time.

# Load vocabularies.
vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)



#sess, model, enc_vocab, rev_dec_vocab = execute.init_session(sess, conf='seq2seq_serve.ini')
#_________________________________________________________________

# start app
if (__name__ == "__main__"): 
    app.run(port = 5005,debug=True,host='0.0.0.0') 
