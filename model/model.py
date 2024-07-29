# using the concept of decision-tree

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
import csv
import os
# import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

training_csv_path = os.path.join(os.path.dirname(__file__), "Training.csv")
testing_csv_path = os.path.join(os.path.dirname(__file__), "Testing.csv")
doc_consult_csv_path = os.path.join(os.path.dirname(__file__), "doc_consult.csv")

# Importing the dataset
training_data = pd.read_csv(training_csv_path)
testing_data = pd.read_csv(testing_csv_path)

# Extracting the feature columns
feature_columns = training_data.columns[:-1]

# Splitting the dataset into features (X) and target (y)
X_train_data = training_data[feature_columns]
y_train_data = training_data["prognosis"]

# Dimensionality reduction by grouping and taking maximum values
reduced_training_data = training_data.groupby(training_data["prognosis"]).max()

# Encoding the target variable to integer values
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y_train_data)
y_train_encoded = label_encoder.transform(y_train_data)

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_train_data, y_train_encoded, test_size=0.33, random_state=42
)

# Preparing the testing data
X_test_data = testing_data[feature_columns]
y_test_data = testing_data["prognosis"]
y_test_encoded = label_encoder.transform(y_test_data)


def get_decision_tree():
    # Implementing the Decision Tree Classifier
    X_train, _, y_train, _ = train_test_split(
        X_train_data, y_train_encoded, test_size=0.33, random_state=42
    )
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree = decision_tree_classifier.fit(X_train, y_train)

    return decision_tree


decision_tree = get_decision_tree()

# Checking the important features
feature_importances = decision_tree.feature_importances_
sorted_feature_indices = np.argsort(feature_importances)[::-1]


def get_disease_name(node):
    """Return the disease name corresponding to a node"""
    node = node[0]
    non_zero_indices = node.nonzero()
    disease = label_encoder.inverse_transform(non_zero_indices[0])
    return disease


def recursive_decision_tree_to_code(user_data, user_input):
    """Generate diagnostic questions and handle user responses"""
    tree = user_data["decision_tree"]
    feature_names = feature_columns

    tree_structure = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_structure.feature
    ]
    symptoms_present = user_data["symptoms_present"]
    node = user_data["node"]
    depth = user_data["depth"]

    if tree_structure.feature[node] != _tree.TREE_UNDEFINED:
        symptom_name = feature_name[node]
        threshold = tree_structure.threshold[node]

        if user_input == "":
            return {"type": "question", "data": symptom_name}

        user_response = user_input.lower()
        if user_response == "yes":
            response_value = 1
        elif user_response == "no":
            response_value = 0

        if response_value <= threshold:
            user_data["node"] = tree_structure.children_left[node]
            user_data["depth"] = depth + 1
        else:
            symptoms_present.append(symptom_name)
            user_data["node"] = tree_structure.children_right[node]
            user_data["depth"] = depth + 1

        return recursive_decision_tree_to_code(user_data, "")
    else:
        diagnosed_disease = get_disease_name(tree_structure.value[node])
        relevant_columns = reduced_training_data.columns
        symptoms_given = relevant_columns[
            reduced_training_data.loc[diagnosed_disease].values[0].nonzero()
        ]
        symptom_indices = np.arange(1, len(symptoms_given) + 1)
        symptom_data = pd.DataFrame(
            list(symptoms_given),
            index=[symptom_indices],
            columns=["OTHER SYMPTOMS"],
        )
        for disease in diagnosed_disease:
            disease_name = disease

        with open(doc_consult_csv_path, "r") as file:
            reader = csv.reader(file)
            consultation_risks = {row[0]: int(row[1]) for row in reader}

        consult_threshold = consultation_risks.get(disease_name, 0)
        should_consult = consult_threshold > 50
        return {
            "type": "diagnosis",
            "diseases": [d for d in diagnosed_disease],
            "disease_name": disease_name,
            "recorded_symptoms": symptoms_present,
            "other_symptoms": symptom_data["OTHER SYMPTOMS"].tolist(),
            "consult_msg": (
                "You should consult a doctor as soon as possible"
                if should_consult
                else "You may consult a doctor for further evaluation"
            ),
        }

user_list = {}


def main(user_id, user_input):
    if user_id not in user_list:
        user_list[user_id] = {
            "decision_tree": get_decision_tree(),
            "symptoms_present": [],
            "node": 0,
            "depth": 1,
        }

    result = recursive_decision_tree_to_code(user_list[user_id], user_input)

    if result["type"] == "diagnosis":
        del user_list[user_id]

    return result
