import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download("stopwords", quiet=True)

# Load stopwords set
stop_words = set(stopwords.words("english"))
