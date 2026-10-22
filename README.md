![Plots Screenshot](images/Screenshot%202026-10-22%20184833.png)
# 🧠 RNN Next-Word Prediction

> A beginner-friendly NLP project that trains a Recurrent Neural Network
> (RNN) to predict the next word in a text sequence.

## 🎯 Project Goal

Build a word-level next-word prediction model using text extracted from
a PDF. Given a short sequence of words, the model predicts the most
likely next word and can generate multiple words iteratively.

## ⚙️ Workflow

1.  **Extract text** from a PDF using PyMuPDF (`fitz`).
2.  **Prepare and split** the text into training, validation, and test
    sets.
3.  **Tokenize** words into integer IDs.
4.  **Create sequences** with 5 input words and the following word as
    the target.
5.  **Build and train** the RNN model.
6.  **Evaluate** training and validation loss/accuracy.
7.  **Generate text** by repeatedly predicting and appending the next
    word.
8.  **Save the model and tokenizer** for reuse.

## 🧰 Technologies & Techniques

-   **Python, NumPy** --- data and numerical operations
-   **PyMuPDF (`fitz`)** --- PDF text extraction
-   **TensorFlow / Keras** --- model construction and training
-   **Tokenizer** --- maps words to token IDs
-   **Embedding (`output_dim=64`)** --- learns word representations
-   **SimpleRNN (128 units)** --- learns sequential patterns
-   **Dropout (0.2)** --- helps reduce overfitting
-   **Dense + Softmax** --- predicts probabilities over the vocabulary
-   **Adam + sparse categorical cross-entropy** --- optimization and
    loss
-   **EarlyStopping / ReduceLROnPlateau** --- optional training
    callbacks
-   **GPU acceleration** --- TensorFlow configured with an NVIDIA RTX
    4050 through WSL2

## 🏗️ Model Architecture

``` python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,
        mask_zero=True
    ),
    tf.keras.layers.SimpleRNN(128),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(vocab_size, activation="softmax")
])
```

**Why five input words?** The model was trained with a context window of
five tokens to predict the next word. Training and prediction must use
the same sequence length.

## 📈 Training

-   Maximum configured training: **30 epochs**
-   Batch size: **32**
-   Validation: separate `(X_val, y_val)` data
-   Metrics: **accuracy** and **loss**

``` python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

> **Tip:** To activate the callbacks, pass
> `callbacks=[reduce_lr, early_stopping]` to `model.fit()`. Defining
> callbacks alone does not enable them.

## ✍️ Generate Multiple Words

The generation function predicts one word, appends it to the text, then
predicts again using the latest five tokens.

``` python
seed_text = "He was a big, beefy man with"
generated_text = predict_next_words(seed_text, num_words=10)
print(generated_text)
```

Generation uses greedy decoding (`argmax`), selecting the
highest-probability word at each step. Errors may accumulate, and longer
outputs can become repetitive or grammatically imperfect.

## 💾 Save & Load

Save the trained model:

``` python
model.save("next_word_rnn.keras")
```

Save the tokenizer because it contains the word-to-ID mapping needed for
prediction:

``` python
import pickle

with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
```

Load them later:

``` python
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("next_word_rnn.keras")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)
```

Keep both files together and reuse the same preprocessing and sequence
length.

## 🧩 Challenges & Lessons Learned

-   **Low prediction accuracy:** Next-word prediction is difficult with
    large vocabularies and rare words.
-   **Data size and quality:** More text does not automatically improve
    accuracy; preprocessing and vocabulary coverage matter.
-   **Overfitting:** Compare validation metrics with training metrics.
-   **Long training runs:** Use reasonable epoch limits and enable
    callbacks when appropriate.
-   **GPU setup:** Verified TensorFlow detects the NVIDIA RTX 4050 in
    WSL2. XLA/cuDNN initialization messages are normal; compiler
    warnings do not necessarily mean training failed.

## ▶️ Run the Project

1.  Open the notebook in VS Code connected to **WSL Ubuntu**.

2.  Select the `tensorflow_gpu` Jupyter kernel.

3.  Install dependencies in that environment:

    ``` bash
    pip install tensorflow numpy pymupdf
    ```

4.  Run notebook cells in order: extraction → preprocessing → sequences
    → model → training → prediction.

5.  Save the model and tokenizer after training.

## 🚀 Future Improvements

-   Compare `SimpleRNN` with **LSTM** or **GRU**.
-   Try cleaner or more representative text and track validation
    performance.
-   Experiment with sequence length, embedding size, and recurrent
    units.
-   Try temperature or top-k sampling for more varied output.

------------------------------------------------------------------------

⭐ **Project takeaway:** This project covers the NLP workflow from PDF
text extraction and tokenization to RNN training, iterative text
generation, and saving the trained model.
