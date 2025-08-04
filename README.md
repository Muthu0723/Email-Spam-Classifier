# Email-Spam-Classifier
A simple and interactive web application that detects whether an email message is Spam or Not Spam (Ham) using Natural Language Processing (NLP) and Machine Learning (ML).

Built with:

🐍 Python

📊 Scikit-learn

🧠 TF-IDF + Naive Bayes

🧹 NLTK for text preprocessing

🖥️ Streamlit for the web interface

🚀 Features
✅ Predict spam/ham from:

Manually typed email text

Uploaded .txt files

Uploaded .eml email files (full email format)

✅ Text preprocessing using:

Lowercasing, punctuation removal

Stopword filtering (via NLTK)

TF-IDF vectorization

✅ Trained on SMS Spam Collection Dataset

🗂️ Project Structure
bash
Copy
Edit
email_spam_classifier/
│
├── app.py                    # Streamlit web app
├── preprocess_train.py       # Data preprocessing and model training
├── requirements.txt          # Python dependencies
│
├── model/
│   ├── spam_classifier.pkl   # Trained Naive Bayes model
│   └── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer
│
└── data/
    └── spam.csv              # Dataset (downloaded from Kaggle)
📦 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download dataset:

Get the dataset from Kaggle - SMS Spam Collection

Rename it to spam.csv and place it inside the data/ folder

🧠 Train the Model
bash
Copy
Edit
python preprocess_train.py
This script:

Cleans and vectorizes the text

Trains a Naive Bayes model

Saves the model and vectorizer in the model/ directory

💻 Run the App
bash
Copy
Edit
streamlit run app.py
Then open http://localhost:8501 in your browser.

🧪 Testing
Test the classifier with:

Free text input like:

vbnet
Copy
Edit
Congratulations! You've won a $1000 Walmart gift card. Click here to claim.
Upload .txt or .eml files

📈 Possible Improvements
Use a larger dataset (e.g., Enron Email Corpus)

Train using other models like SVM or XGBoost

Add prediction confidence

Enable batch email classification

📝 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Muthu Nivesh
Built using Streamlit, Scikit-learn, NLTK
Feel free to connect and contribute!
