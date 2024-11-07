# The dataset has 3 types of labels - toxic, severe_toxic, non_toxic.
# The objective is to build a classifier that can predict if a comment is toxic, severe_toxic or non_toxic. 
# This is a binary classification problem.

import pandas as pd # for data manipulation and analysis            
import numpy as np # for numerical operations   
from llama_index.embeddings.ollama import OllamaEmbedding # for embedding generation
from tqdm import tqdm # for progress bar
from sklearn.ensemble import RandomForestClassifier # for classification
from sklearn.metrics import roc_auc_score # for evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # for evaluation
import pickle # for model persistence   


# Toxic Rail class
# This class is used to train and predict toxicity in comments using embeddings and a Random Forest classifier.
class Toxic_Rail:
    """
    A class to handle training and prediction of toxicity in comments using embeddings and a Random Forest classifier.

    Attributes:
    embed_model (OllamaEmbedding): The embedding model used for generating text embeddings.
    model (RandomForestClassifier): The trained Random Forest model for prediction.
    df_train (DataFrame): The prepared training data.
    """

    # constructor
    # This method initializes the Toxic_Rail class with either training or prediction mode. 
    def __init__(self, mode, input_path=None): # constructor
        """
        Initializes the Toxic_Rail class with either training or prediction mode.

        Args:
        mode (str): The mode of operation, either "train" or "predict".
        input_path (str, optional): The path to the input CSV file for training.
        """
        self.embed_model = OllamaEmbedding(model_name="mxbai-embed-large:latest") # embedding model
        self.X = pd.DataFrame([])
        if mode == "train": # train mode
            self.X, self.model = self.train(input_path, "model/clf.mdl")
        if mode == "predict": # predict mode
            self.model = pickle.load(open("model/clf.mdl", "rb"))
        if mode == "evaluate": # evaluate mode
            self.model = pickle.load(open("model/clf.mdl", "rb"))
            self.evaluate(input_path)

    # evaluate method
    # This method evaluates the model on a test dataset and returns the accuracy and AUC score. 
    def evaluate(self, test_path): 
        tqdm.pandas()
        df_test = self.prepare_data(test_path)
        embeddings = df_test['comment_text'].progress_apply(self.get_embeddings) # get embeddings   
        columns = ["embed_" + str(i) for i in range(len(embeddings[0]))] # column names
        X_test = pd.DataFrame(embeddings.tolist(), columns=columns) # convert embeddings to dataframe
        y_test = df_test['label'] # get labels
        score = self.model.score(X_test, y_test) # get accuracy
        auc_val = roc_auc_score(y_test, self.model.predict_proba(X_test)[:,1]) # get AUC score
        return score, auc_val
        
    # train method
    # This method trains the Random Forest model on the provided dataset and saves the model.   
    def train(self, input_path, model_persist_path): 
        """
        Trains the Random Forest model on the provided dataset and saves the model.

        Args:
        input_path (str): The path to the input CSV file for training.
        model_persist_path (str): The path where the trained model will be saved.
        """
        tqdm.pandas()
        self.df_train = self.prepare_data(input_path)
        embeddings = self.df_train['comment_text'].progress_apply(self.get_embeddings)
        columns = ["embed_" + str(i) for i in range(len(embeddings[0]))]
        X = pd.DataFrame(embeddings.tolist(), columns=columns)
        y = self.df_train['label']
        clf = RandomForestClassifier(max_depth=16, random_state=0).fit(X, y)
        print(clf.score(X, y))
        pickle.dump(clf, open(model_persist_path, "wb"))
        return X, clf

    # predict method
    # This method predicts the toxicity label for a given text using the trained model. 
    def predict(self, text): 
        """
        Predicts the toxicity label for a given text using the trained model.

        Args:
        text (str): The text to be classified.

        Returns:
        array: The predicted label for the input text.
        """
        test_embed = self.embed_model.get_text_embedding(text)
        column_names = ["embed_" + str(i) for i in range(len(test_embed))]
        X_test = pd.DataFrame([test_embed], columns=column_names)
        return self.model.predict(X_test)

    # create label method
    # This method creates a combined label for a row of the dataset based on individual toxicity indicators.    
    def create_label(self, row): 
        """
        Creates a combined label for a row of the dataset based on individual toxicity indicators.

        Args:
        row (Series): A row from the dataset.

        Returns:
        str: The combined label for the row.
        """
        label = "|"
        if row["toxic"] == 1:
            label += "toxic|"
        if row["severe_toxic"] == 1:
            label += "severe_toxic|"
        if row["obscene"] == 1:
            label += "obscene|"
        if row["threat"] == 1:
            label += "threat|"
        if row["insult"] == 1:
            label += "insult|"
        if row["identity_hate"] == 1:
            label += "identity_hate|"
        if label == "|":
            label = "|non_toxic|"
        return label

    # get embeddings method
    # This method generates embeddings for a given text using the embedding model.  
    def get_embeddings(self, text): 
        """
        Generates embeddings for a given text using the embedding model.

        Args:
        text (str): The text to be embedded.

        Returns:
        list: The embedding of the text.
        """
        return self.embed_model.get_text_embedding(text=text)

    def prepare_data(self, input_path): 
        """
        Prepares the training data by reading from a CSV file, creating labels, 
        and balancing the dataset. The original dataset from here - 
        https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

        Args:
        input_path (str): The path to the input CSV file.

        Returns:
        DataFrame: The prepared training data.
        """
        df = pd.read_csv(input_path)
        df["label"] = df.progress_apply(self.create_label, axis=1)
        toxic_df = df.iloc[np.where(df["label"] != "|non_toxic|")]
        non_toxic_df = df.iloc[np.where(df["label"] == "|non_toxic|")].head(10000)
        df = pd.concat([toxic_df, non_toxic_df], ignore_index=True, sort=False)
        return df