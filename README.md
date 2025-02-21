This project performs sentiment analysis on restaurant reviews to classify them as positive or negative using a Logistic Regression model. The model processes customer reviews by applying text preprocessing, TF-IDF vectorization, and classification techniques.
📂 Dataset Information

The dataset (Restaurant_Reviews.tsv) contains customer reviews and sentiment labels:
Review (Review) – Text feedback from customers
Liked (Liked) – Sentiment label (1 = Positive, 0 = Negative)
🛠 Technologies & Libraries Used

Programming Language: Python
Libraries:
pandas – Data handling
numpy – Numerical computations
nltk – Natural Language Processing
scikit-learn – Machine Learning
📊 Model & Approach

1️⃣ Preprocessing
Text Cleaning – Lowercased text, removed stopwords
Tokenization – Used nltk.word_tokenize()
Stemming – Applied PorterStemmer()
TF-IDF Vectorization – Converted text into numerical features
2️⃣ Model Training
Classifier: Logistic Regression (solver=liblinear, C=1)
Hyperparameter Tuning: Used GridSearchCV to find the best parameters
3️⃣ Training & Evaluation
Train-Test Split: 80% Training, 20% Testing
Accuracy Achieved: 81.50%
4️⃣ Prediction Example
Given input "It is not so good", the model predicted:
👍 This is a positive review!
🚀 How to Run the Project

Install required libraries (if not installed):
pip install pandas numpy nltk scikit-learn
Run the Python script:
python sentimental_analysis.py
Expected Output Example:
Accuracy: 81.50%
Best parameters: {'C': 1, 'solver': 'liblinear'}
👍 This is a positive review!
📌 Future Improvements

✅ Use Word2Vec or BERT embeddings for better feature extraction
✅ Implement deep learning models (LSTMs, Transformers) for improved accuracy
✅ Deploy the model as a web application using Flask or Streamlit
📜 Author

Developed by [Your Name] (replace with your GitHub/LinkedIn if needed