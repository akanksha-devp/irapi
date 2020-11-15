from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd 
import pickle
import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
# import joblib

# from sklearn.externals import joblib
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    RANDOM_SEED = 42

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    train = pd.read_csv("HT_train.csv")
    valid = pd.read_csv("HT_valid.csv")
    test = pd.read_csv("HT_test.csv")
    train = train.append(valid).reset_index(drop=True)
    bert_model_name="intent"

    bert_ckpt_dir = os.path.join("model2/", bert_model_name)    #checkpoint directory
    bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt") #checkpoint file
    bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")  #configuration file
    class IntentDetectionData:
        DATA_COLUMN = "text"
        LABEL_COLUMN = "intent"
        def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
            self.tokenizer = tokenizer
            self.max_seq_len = 0
            self.classes = classes
    
            train, test = map(lambda df: df.reindex(df[IntentDetectionData.DATA_COLUMN].str.len().sort_values().index), [train, test])
    
            ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

            print("max seq_len", self.max_seq_len)
            self.max_seq_len = min(self.max_seq_len, max_seq_len)
            self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])
        def _prepare(self, df):
            x, y = [], []
    
            for _, row in tqdm(df.iterrows()):
                text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len, len(token_ids))
                x.append(token_ids)      
                y.append(self.classes.index(label))
            return np.array(x), np.array(y)
        def _pad(self, ids):
            x = []
            for input_ids in ids:
                input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
                input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
                x.append(np.array(input_ids))
            return np.array(x)
    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
    def create_model(max_seq_len, bert_ckpt_file):
        with tf.io.gfile.GFile(bert_config_file, "r") as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = None
            bert = BertModelLayer.from_params(bert_params, name="bert")
        input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
        bert_output = bert(input_ids)

        print("bert shape", bert_output.shape)

        cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
        cls_out = keras.layers.Dropout(0.5)(cls_out)
        logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
        logits = keras.layers.Dropout(0.5)(logits)
        logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

        model = keras.Model(inputs=input_ids, outputs=logits)
        model.build(input_shape=(None, max_seq_len))

        load_stock_weights(bert, bert_ckpt_file)
                
        return model
    classes = train.intent.unique().tolist()

    data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=128)
    model = create_model(data.max_seq_len, bert_ckpt_file)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    history = model.fit(
        x=data.train_x, 
        y=data.train_y,
        validation_split=0.1,
        batch_size=16,
        shuffle=True,
        epochs=5,
#   callbacks=[tensorboard_callback]
    )
    model.save("save_model")
    model=keras.models.load_model('save_model')
    # saved_model = pickle.dumps(model) 
  
    # # Load the pickled model 
    # knn_from_pickle = pickle.loads(saved_model) 
      
    # Use the loaded pickled model to make predictions 
    # knn_from_pickle.predict(X_test) 

    # joblib.dump(model, 'intent_model.pkl')
    # # intent_model = open('intent_model.pkl','rb')
    # model20 = joblib.load('intent_model.pkl')
    # with open('model_pickle','wb') as f:
    #     pickle.dump(model,f)
    # with open('model_pickle','rb') as f:
    #     mp=pickle.load(f)

    if request.method == 'POST':
        message = request.form['message']
        inputdata = [message]
        pred_tokens = map(tokenizer.tokenize, inputdata)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

        pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))

        
        my_prediction = model.predict(pred_token_ids).argmax(axis=-1)
        if my_prediction==0:
            x="Details"
        elif my_prediction==1:
            x="Create"
        elif my_prediction==2:
            x="Book"
    # return jsonify(inputdata)
    # return jsonify(my_prediction)
    return render_template('result.html',prediction = x)
    # return render_template('home.html', my_prediction)
     


if __name__ == '__main__':
    app.run(debug=True)