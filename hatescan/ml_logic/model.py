# Import modules
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.models import Sequential, layers, regularizers, optimizers
from tensorflow.keras.preprocessing.text import text_to_word_sequence

#create attention class
class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros")
        super(Attention, self).build(input_shape)


    def call(self, x):
        e = K.tanh(K.dot(x,self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        return K.sum(output, axis=1)

    def get_config(self):
        return super(Attention, self).get_config()



#import, initialize and compile model

def initialize_model(vocab_size, embedding_dimension, learning_rate):
    l2 = regularizers.L2() #play with hyperparams
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dimension, mask_zero=True))
    model.add(layers.Masking())
    model.add(Bidirectional(LSTM(32, kernel_regulizer=l2)))
    model.add(Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2)))
    model.add(Attention()) #add attention layer
    model.add(Dense(16, activation='relu', kernel_reguralizer=l2))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer , metrics=['Precision', 'Accuracy']) #precision because unbalanced dataset

    return model


# Fitting the model
def train_model(
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size=64,
        patience=10,
        validation_split=0.2
    ):

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
        class_weight = {0: (1-(35193/51842)), 1: (1-(9221/51842)), 2: (1-(7428/51842))} #more weight on class 1 & 2
    )

    return model, history

# Evaluating the model
def evaluate_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ):
    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    test_loss, test_precision, test_acc = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        return_dict=True
    )

    print(f"✅ Model evaluated, Precision: {round(test_precision, 2)}")
    print(f"✅ Model evaluated, Accuracy: {round(test_acc, 2)}")

    return test_loss, test_precision, test_acc

# Making predictions fromt the model
def model_predict(model,
                  X_new: np.array):

    predictions = model.predict(X_new)
    print(predictions)


##XGBoost Classifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import xgboost as xgb

# Feature/Target
X_new = df["Cleaned_text"]
y_new = df[['HateLabel']]

#tokenizing
vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1,3))
X_vectorized = vectorizer.fit_transform(X_new)

# Calculate explained variance ratio and cumulative explained variance
pca=PCA()
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot cumulative explained variance to set max_features (at 80%)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', linewidth=2)
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid(True)
plt.show()

#instantiate pca with n_components found above
pca = PCA(n_components=1700)
X_pca = pca.fit_transform(X_vectorized.toarray())
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_new)

#instantiate xgboost
xgb_reg = xgb.XGBClassifier(max_depth=10, n_estimators=100)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1', 'Class 2'])
print(report)

## IMPORTED MODEL FROM ROBERRTAAA
# Imported model from HuggingFace

import requests

API_TOKEN = "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" #Mauro's token
API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-tweets-hate-speech"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
	"inputs": input_tweet
})

#return class prediction
def return_class(output):
  if 'LABEL_0' in output[0][0]['label']:
    if output[0][0]["score"] > 0.5:
      print("class 2: NEUTRAL")

    elif output[0][0]["score"] < 0.3:
      print("class 1: OFFENSIVE")

    else:
      print("class 0: HATE")

  elif 'LABEL_1' in output[0][0]['label']:
    if output[0][0]["score"] > 0.7:
      print("class 2: HATE")

    elif output[0][0]["score"] < 0.5:
      print("class 1: OFFENSIVE")

    else:
      print("class 0: NEUTRAL")
