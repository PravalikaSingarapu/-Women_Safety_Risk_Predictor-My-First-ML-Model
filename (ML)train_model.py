import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Placeholder for your actual data loading and preprocessing logic
# Your existing logic to create the sample data

# Make sure to save your model and encoders after training
# Example of saving model
with open('model/safety_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer, label_encoders, target_le), f)
