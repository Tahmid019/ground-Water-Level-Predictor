import pandas as pd
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

class LogisticModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X = None
        self.Y = None
        self.model = LogisticRegression(random_state=0)
        self.scaler = StandardScaler()
        
    def preprocess(self):
        sb.countplot(x='Situation', data=self.data, palette='bright')
        
        labelencoder_Y = LabelEncoder()
        self.Y = labelencoder_Y.fit_transform(self.data.iloc[:, 12])
        
        self.X = self.data.iloc[:, [4, 5, 6, 9, 10, 11]]
        
    def handle_categorical(self):
        availability = pd.get_dummies(self.data['Situation'], drop_first=True)
        self.data.drop(['Situation'], axis=1, inplace=True)
        self.data = pd.concat([self.data, availability], axis=1)
        self.X = self.data.iloc[:, [4, 5, 6, 9, 10, 11]]
        self.Y = self.data.iloc[:, 12]
    
    def split_data(self, test_size=1/3):
        return train_test_split(self.X, self.Y, test_size=test_size, random_state=0)
    
    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
        
    def evaluate_model(self, X_test, Y_test):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(Y_test, y_pred)
        accuracy = accuracy_score(Y_test, y_pred)
        return cm, accuracy

    def cross_validate(self, cv_folds=5):
        print("Starting Cross Validation...")
        scores = cross_val_score(self.model, self.X, self.Y, cv=cv_folds, scoring='accuracy')
        for i in tqdm(range(cv_folds)):
            pass
        return scores
