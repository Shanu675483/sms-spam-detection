# sms-spam-detection
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# 2. Clean the dataset by selecting only the required columns and renaming them
df = df[['v1', 'v2']]  # Keep only the necessary columns
df.columns = ['label', 'message']  # Rename the columns

# 3. Encode the labels (convert 'ham' to 0 and 'spam' to 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# 5. Define a text preprocessing function to clean the messages
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize the text into words
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)

# Apply the cleaning function to each message
df['message'] = df['message'].apply(clean_text)

# 6. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 7. Convert the text data into numerical form using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 8. Train the Naïve Bayes classifier on the training data
model = MultinomialNB()
model.fit(X_train, y_train)

# 9. Predict the labels for the test data
y_pred = model.predict(X_test)

# 10. Evaluate and print the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
