import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

news = pd.read_csv("data/News.csv")



#print(fake.shape)
#print(true.shape)

# Add flag to track fake and real
#fake['target'] = 'FAKE'
#true['target'] = 'REAL'

# Concatenate dataframes
#data = pd.concat([fake, true]).reset_index(drop = True)


data = news
print(data.shape)

# Shuffle the data
def Shuffle(data):
    from sklearn.utils import shuffle
    data = shuffle(data)
    data = data.reset_index(drop=True)
    # Check the data
    print("Check the data")
    print(data.head())
    print("Shuffled")


def Remove_Head_Col(data, col_name):
    data.drop([col_name],axis=1,inplace=True)
    print("Removing the "+ col_name)
    print(data.head())
    
def Convert_the_case(data, col_name, case):
    if(case == "lower"):
        data[col_name] = data[col_name].apply(lambda x: x.lower())
        print("Convert to "+case+"case")
    elif(case == "upper"):
        data[col_name] = data[col_name].apply(lambda x: x.upper())
        print("Convert to "+case+"case")
    else:
        print("Wrong typed case: "+case)
    print(data.head())

Shuffle(data)

#Remove_Head_Col(data, "date")
Remove_Head_Col(data, "title")

Convert_the_case(data, 'text', "lower")


# Remove punctuation
print("Remove punctuation")
import string

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)


# Check
print("Check")
print(data.head())
print("punctuation removed")

# Removing stopwords
print("Removing stopwords")
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Check
print("Check")
print(data.head())
print("stopwords removed")




#Basic data exploration

# How many fake and real articles?
def fake_and_real_articles():
    print("How many fake and real articles")
    print(data.groupby(['target'])['text'].count())
    data.groupby(['target'])['text'].count().plot(kind="bar")
    plt.show()


# Word cloud for fake news
def Word_cloud_for_fake_news():
    print("Word cloud for fake news")
    from wordcloud import WordCloud

    fake_data = data[data["target"] == "FAKE"]
    all_words = ' '.join([text for text in fake_data.text])

    wordcloud = WordCloud(width= 800, height= 500,
                              max_font_size = 110,
                              collocations = False).generate(all_words)

    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# Word cloud for real news
def Word_cloud_for_real_news():
    print("Word cloud for real news")
    from wordcloud import WordCloud

    real_data = data[data["target"] == "REAL"]
    all_words = ' '.join([text for text in fake_data.text])

    wordcloud = WordCloud(width= 800, height= 500,
                              max_font_size = 110,
                              collocations = False).generate(all_words)

    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()



# Most frequent words counter
from nltk import tokenize

token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    print("Most frequent words counter")
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()





# Most frequent words in fake news
#counter(data[data["target"] == "FAKE"], "text", 20)


# Most frequent words in real news
#counter(data[data["target"] == "REAL"], "text", 20)

# Word cloud for real news
#Word_cloud_for_real_news()


# Word cloud for fake news
#Word_cloud_for_fake_news()

# How many fake and real articles?
#fake_and_real_articles()




#MODELING

# Function to plot the confusion matrix
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print("Function to plot the confusion matrix")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



#Peparing the data
# Split the data
print("Peparing the data")
X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)




#Logistic regression
from sklearn.linear_model import LogisticRegression
def Logistic_regression():
    # Vectorizing and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', LogisticRegression())])
    # Fitting the model
    model = pipe.fit(X_train, y_train)
    # Accuracy
    prediction = model.predict(X_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

    cm = metrics.confusion_matrix(y_test, prediction)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'])




#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
def Decision_Tree_Classifier():
    # Vectorizing and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', DecisionTreeClassifier(criterion= 'entropy',
                                               max_depth = 20,
                                               splitter='best',
                                               random_state=42))])
    # Fitting the model
    model = pipe.fit(X_train, y_train)
    # Accuracy
    prediction = model.predict(X_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

    cm = metrics.confusion_matrix(y_test, prediction)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'])





#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
def Random_Forest_Classifier():
    # Vectorizing and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
    # Fitting the model
    model = pipe.fit(X_train, y_train)
    # Accuracy
    prediction = model.predict(X_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

    cm = metrics.confusion_matrix(y_test, prediction)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'])



#Logistic_regression()

#Decision_Tree_Classifier()

Random_Forest_Classifier()
