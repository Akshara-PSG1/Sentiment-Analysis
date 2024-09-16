with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess(text):
  sequences = tokenizer.texts_to_sequences([text])
  padded_sequences = pad_sequences(sequences, maxlen = 250)
  return padded_sequences

def predict(text):
  processed_text = preprocess(text)
  prediction = model.predict(processed_text)
  return "Positive" if prediction[0][0] > 0.5 else "Negative"

gradio = gd.Interface(
    fn = predict,
    inputs = gd.Textbox(lines = 2, placeholder = 'Enter a sentence here...'),
    outputs = 'text',
    title = "Sentiment Analysis",
    description = "Enter a sentence to predict its sentiment. The model will predict it as Positive or Negative ")

gradio.launch()
