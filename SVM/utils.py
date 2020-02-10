import numpy as np
import math
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from svm import SVM, gaussian_kernel


def get_vocab_list(path):
    vocab_list = dict()

    with open(path, 'r') as f:
        for vocabs in f:
            i, vocab = vocabs.strip().split('\t')
            vocab_list[vocab] = int(i)
    return vocab_list


def get_vocab_list2(path):
    vocab_list = dict()

    with open(path, 'r') as f:
        for vocabs in f:
            i, vocab = vocabs.strip().split('\t')
            vocab_list[int(i)] = vocab
    return vocab_list


def process_email(email_contents, vocab_list):
    """Preprocess the body of an email and returns a list of word_indices
    """
    word_indices = []
    # Lower case
    email_contents = email_contents.lower().strip()
    # Strip the HTL
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    # Handle numbers
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    # Handle urls
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    # Get rid of punctuation and Tokenize words
    replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    remove_punct = email_contents.translate(replace_punctuation)
    # Remove stop words
    stop_words = list(stopwords.words('english'))
    additional_stopwords = ["'s", "...", "'ve", "``", "''", "'m", '--', "'ll", "'d"]
    stop_words = set(stop_words + additional_stopwords)
    word_tokens = word_tokenize(remove_punct)
    # word_tokens = [w for w in word_tokens if not w in stop_words]
    # Lematization
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in word_tokens]
    for word in stemmed:
        if word in vocab_list.keys():
            word_indices.append(int(vocab_list[word]))
    return word_indices


def email_features(word_indices, num_features):
    x = np.zeros(num_features)
    for index in range(len(word_indices)):
        x[word_indices[index] - 1] = 1
    return np.reshape(x, (num_features, 1), order='F')


def parameter_search(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    param_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 5, 10, 30]
    init_error = math.inf

    for i in param_vec:
        c = i
        for j in param_vec:
            s = j
            model = SVM(kernel=gaussian_kernel, C=c, sigma=sigma)
            model.svmTrain(X, y)
            predictions = model.svmPredict(Xval)
            error = np.mean(predictions != yval)
            if error < init_error:
                init_error = error
                C = c
                sigma = s
    return C, sigma


def plot_data(features, labels, ax):
    labels = labels.flatten()
    pos = np.where(labels == 1)
    neg = np.where(labels == 0)

    ax.scatter(features[pos, 0], features[pos, 1], c='b', marker='o')
    ax.scatter(features[neg, 0], features[neg, 1], c='r', marker='x')


def plot_linear_decision_boundary(X, y, model, ax):
    w = model.model['w']
    b = model.model.get('b')
    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yp = - (w[0] * xp + b) / w[1]
    plot_data(X, y, ax)
    ax.plot(xp, yp, color='blue')
