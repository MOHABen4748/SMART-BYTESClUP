Fake News Detection (Arabic + English)

This project is a machine learning web application that detects whether a news article is Real or Fake.
The system accepts Arabic or English text:

If the input is Arabic, it is automatically translated to English.

The text is processed using TF-IDF.

A Logistic Regression classifier predicts Fake or Real.

The model is trained on an English Fake/Real news dataset.

🚀 Features

Detect Fake News in Arabic & English

Automatic translation (Arabic → English)

Arabic text normalization & cleaning

TF-IDF vectorization

Logistic Regression classifier

Flask web interface

Beautiful modern UI (HTML + CSS)

Model + Vectorizer saved for reuse

📁 Project Structure
news-detection/
│
├── app.py
├── train.py
├── model_utils.py
│
├── Dataset/
│   ├── true.csv
│   └── fake.csv
│
├── saved/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   └── index.html
│
└── static/
    └── css/
        └── style.css

⚙️ Installation
1️⃣ Create a virtual environment
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


macOS/Linux

source venv/bin/activate

2️⃣ Install dependencies

If you have a requirements file:

pip install -r requirements.txt


Or install manually:

pip install flask scikit-learn pandas numpy deep-translator

🧠 Train the Machine-Learning Model

Run:

python train.py


This will:

Load the dataset from Dataset/

Clean and normalize text

Extract TF-IDF features

Train the model

Save model.pkl and vectorizer.pkl into /saved

You should see:

Model trained successfully!
Model + Vectorizer Saved Successfully!

▶️ Run the Web Application

Start the Flask server:

python app.py


Then open the app in your browser:

http://127.0.0.1:5000/

🎨 User Interface

The interface includes:

A big text box for user input

A modern blue “Analyze” button

A result box showing Fake or Real

Clean and responsive design

Built using:

HTML5

CSS3

Flask Jinja templates

📌 Technologies Used

Python 3.10+

Flask (web app)

Scikit-Learn (ML model)

Pandas / NumPy (data processing)

Deep Translator (Arabic → English)

Logistic Regression

TF-IDF Vectorizer

👨‍💻 Author

Developed by Abdou Dahbi
Open-source & free to use for learning and research.

⭐️ How to Support

If this project helped you:

Give the GitHub repository a ⭐️

Share it with your team or classmates

Fork it and build your own improvements

📄 License

This project is released under the MIT License.
You may use, modify, and distribute it freely.
