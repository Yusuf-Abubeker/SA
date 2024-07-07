from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('./content/sentiment_tokenizer')

# Load the models
model_primary = TFBertForSequenceClassification.from_pretrained('./content/sentiment_model_primary')
model_good = TFBertForSequenceClassification.from_pretrained('./content/sentiment_model_good')
model_bad = TFBertForSequenceClassification.from_pretrained('./content/sentiment_model_bad')

# Load the LabelEncoders
le_primary = joblib.load('./content/le_primary.pkl')
le_good = joblib.load('./content/le_good.pkl')
le_bad = joblib.load('./content/le_bad.pkl')

def predict_category(comment):
    # Tokenize the input comment
    inputs = tokenizer(
        comment,
        max_length=100,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    
    # Predict primary category
    primary_preds = model_primary(inputs['input_ids'])
    primary_category = tf.argmax(primary_preds.logits, axis=1).numpy()[0]
    primary_category_label = le_primary.inverse_transform([primary_category])[0]

    if primary_category_label == 'Good':
        # Predict secondary category for good comment
        secondary_preds = model_good(inputs['input_ids'])
        secondary_category = tf.argmax(secondary_preds.logits, axis=1).numpy()[0]
        secondary_category_label = le_good.inverse_transform([secondary_category])[0]
    else:
        # Predict secondary category for bad comment
        secondary_preds = model_bad(inputs['input_ids'])
        secondary_category = tf.argmax(secondary_preds.logits, axis=1).numpy()[0]
        secondary_category_label = le_bad.inverse_transform([secondary_category])[0]
    
    return primary_category_label, secondary_category_label

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get('comment')
    
    if not comment:
        return jsonify({'error': 'No comment provided'}), 400
    
    primary_category, secondary_category = predict_category(comment)
    
    return jsonify({
        'primary_category': primary_category,
        'secondary_category': secondary_category
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
