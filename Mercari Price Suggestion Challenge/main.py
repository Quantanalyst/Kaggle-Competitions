
"""
Mercari competition (Kaggle)

@author: Saeed Mohajeryami, PhD

"""

import numpy as np
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import collections
import plotly as py
import plotly.graph_objs as go
from plotly.offline import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from wordcloud import WordCloud
import bokeh.plotting as bp
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool


train = pd.read_csv('train.tsv', delimiter = '\t')
test = pd.read_csv('test.tsv', delimiter = '\t')

## downsample 
train = train.sample(frac=0.05, replace=True, random_state=0)
test = test.sample(frac=0.05, replace=True, random_state=0)

# size of training and dataset
print(train.shape)
print(test.shape)

# data types
print(train.dtypes)

# exploration
print(train.columns)
train['item_condition_id'].value_counts()
train['category_name'].value_counts()
train['brand_name'].value_counts()
print("There are %d unique brand names in the training dataset." 
      % train['brand_name'].nunique())

train.price.describe()
## right skewed price distribution
train[train['price'] < 300]['price'].hist()
train['shipping'].value_counts()


## Create three new features based on category name
## Category name, if available, has three parts, which are separated by '/'
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))

## description of new features
print("There are %d unique general-categories." % train['general_cat'].nunique())
print("There are %d unique first sub-categories." % train['subcat_1'].nunique())
print("There are %d unique second sub-categories." % train['subcat_2'].nunique())

## visualization
x = train['general_cat'].value_counts().index.values.astype('str')
y = train['general_cat'].value_counts().values
pct = [("%.2f"%(v*100))+"%"for v in (y/len(train))]

trace1 = go.Bar(x=x, y=y, text=pct)
layout = dict(title= 'Number of Items by Main Category',
              yaxis = dict(title='Count'),
              xaxis = dict(title='Category'))
fig=dict(data=[trace1], layout=layout)
py.offline.plot(fig, filename='general_cat_stat.html')

x = train['subcat_1'].value_counts().index.values.astype('str')[:15]
y = train['subcat_1'].value_counts().values[:15]
pct = [("%.2f"%(v*100))+"%"for v in (y/len(train))][:15]

trace1 = go.Bar(x=x, y=y, text=pct)
layout = dict(title= 'Number of Items by Subcategory 1',
              yaxis = dict(title='Count'),
              xaxis = dict(title='Subcategory'))
fig=dict(data=[trace1], layout=layout)
py.offline.plot(fig, filename='subcat1_stat.html')

## Item Descriptions
## It will be more challenging to parse through this particular item since
## it's unstructured data. Does it mean a more detailed and lengthy description
## will result in a higher bidding price? We will strip out all punctuations,
## remove some english stop words (i.e. redundant words such as "a", "the", etc.)
## and any other words with a length less than 3:

### Natural Language Processing part
stop_words = set(stopwords.words('english'))

def wordCount(text):
    # convert to lower case and strip regex
    try:
         # convert to lower case and strip regex
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        # tokenize
        # words = nltk.word_tokenize(clean_txt)
        # remove words in stop words
        words = [w for w in txt.split(" ") if not w in stop_words and len(w)>3]
        return len(words)
    except: 
        return 0
## notes: re.compile is to create our desired patterns. In this case, our desired
## patten is [!"\#\$%\&\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~0-9\r\t\n]   
        
# add a column of word counts to both the training and test set
train['desc_len'] = train['item_description'].apply(lambda x: wordCount(x))
test['desc_len'] = test['item_description'].apply(lambda x: wordCount(x))

df = train.groupby('desc_len')['price'].mean().reset_index()

## visualizing the relationship b/w description length and price
plt.plot(df.desc_len, df.price)
plt.title('description length vs. price')
plt.xlabel('description length')
plt.ylabel('price $')
plt.show()

# remove missing values in item description
train = train[pd.notnull(train['item_description'])]


## Pre-processing: tokenization
##Most of the time, the first steps of an NLP project is to "tokenize" your
## documents, which main purpose is to normalize our texts. The three fundamental
## stages will usually include:
##  break the descriptions into sentences and then break the sentences into tokens
##  remove punctuation and stop words
##  lowercase the tokens
##  herein, I will also only consider words that have length equal to or greater
## than 3 characters

def tokenize(text):
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try: 
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text) # remove punctuation
        
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        ### tokens_ is a list of lists, we should make only a list
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        ## remove stop words by filtering tokens
        tokens = list(filter(lambda t: t.lower() not in stop_words, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        
        return filtered_tokens
            
    except TypeError as e: print(text,e)
    
# apply the tokenizer into the item descriptipn column
train['tokens'] = train['item_description'].map(tokenize)
test['tokens'] = test['item_description'].map(tokenize)

## check to see if our tokenizer function works correctly
for description, tokens in zip(train['item_description'].head(),
                              train['tokens'].head()):
    print('description:', description)
    print('tokens:', tokens)
    print()
    
# build dictionary with key=category and values as all the descriptions related.
general_cats = list(train.general_cat.unique())
cat_desc = dict()
for cat in general_cats: 
    text = " ".join(train.loc[train['general_cat'] == cat, 'item_description'].values)
    cat_desc[cat] = tokenize(text)


# find the most common words for the top 4 categories
women100 = collections.Counter(cat_desc['Women']).most_common(100)
beauty100 = collections.Counter(cat_desc['Beauty']).most_common(100)
kids100 = collections.Counter(cat_desc['Kids']).most_common(100)
electronics100 = collections.Counter(cat_desc['Electronics']).most_common(100)

## Generate word cloud
def generate_wordcloud(tup):
    wordcloud = WordCloud(background_color='white',
                          max_words=50, max_font_size=40,
                          random_state=42
                         ).generate(str(tup))
    return wordcloud


# Display the generated image - women100:
plt.imshow(generate_wordcloud(women100), interpolation='bilinear')
plt.axis("off")
plt.show()

# Display the generated image - beauty100:
plt.imshow(generate_wordcloud(beauty100), interpolation='bilinear')
plt.axis("off")
plt.show()

## Pre-processing: tf-idf¶
## tf-idf is the acronym for Term Frequency–inverse Document Frequency.
## It quantifies the importance of a particular word in relative to the 
## vocabulary of a collection of documents or corpus. The metric depends
## on two factors:
##
##  Term Frequency: the occurences of a word in a given document
## (i.e. bag of words)
##  Inverse Document Frequency: the reciprocal number of times a word
## occurs in a corpus of documents
## Think about of it this way: If the word is used extensively in all documents,
## its existence within a specific document will not be able to provide us much
## specific information about the document itself. So the second term could be
## seen as a penalty term that penalizes common words such as "a", "the", "and",
## etc. tf-idf can therefore, be seen as a weighting scheme for words relevancy
## in a specific document.

vectorizer = TfidfVectorizer(min_df=10,
                             max_features=180000,
                             tokenizer=tokenize,
                             ngram_range=(1, 2))

all_desc = np.append(train['item_description'].values, test['item_description'].values)
vz = vectorizer.fit_transform(list(all_desc))

## vz is a tfidf matrix where:
##      - the number of rows is the total number of descriptions
##      - the number of columns is the total number of unique tokens across
## the descriptions

#  create a dictionary mapping the tokens to their tfidf values
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
tfidf.columns = ['tfidf']

tfidf.sort_values(by=['tfidf'], ascending=True).head(10)
tfidf.sort_values(by=['tfidf'], ascending=False).head(10)



## Given the high dimension of our tfidf matrix, we need to reduce their 
## dimension using the Singular Value Decomposition (SVD) technique. And to
## visualize our vocabulary, we could next use t-SNE to reduce the dimension
## from 50 to 2. t-SNE is more suitable for dimensionality reduction to 2 or 3.

## t-Distributed Stochastic Neighbor Embedding (t-SNE)
## t-SNE is a technique for dimensionality reduction that is particularly well
## suited for the visualization of high-dimensional datasets. The goal is to take
## a set of points in a high-dimensional space and find a representation of those
## points in a lower-dimensional space, typically the 2D plane. It is based on
## probability distributions with random walk on neighborhood graphs to find the
## structure within the data. But since t-SNE complexity is significantly high,
## usually we'd use other high-dimension reduction techniques before applying
## t-SNE.

## First, let's take a sample from the both training and testing item's
## description since t-SNE can take a very long time to execute. We can then
## reduce the dimension of each vector from to n_components (50) using SVD.

trn = train.copy()
tst = test.copy()
trn['is_train'] = 1
tst['is_train'] = 0

sample_sz = 15000

combined_df = pd.concat([trn, tst], ignore_index = True)
combined_sample = combined_df.sample(n=sample_sz)

vectorizer_sample = TfidfVectorizer(min_df=10,
                             max_features=180000,
                             tokenizer=tokenize,
                             ngram_range=(1, 2))

vz_sample = vectorizer_sample.fit_transform(list(combined_sample['item_description']))

## Dimensionality reduction by SVD 
n_comp=50
svd = TruncatedSVD(n_components=n_comp, random_state=42)
svd_tfidf_sample = svd.fit_transform(vz_sample)

## Now we can reduce the dimension from 50 to 2 using t-SNE!
tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)
tsne_tfidf_sample = tsne_model.fit_transform(svd_tfidf_sample)

## It's now possible to visualize our data points. Note that the deviation
## as well as the size of the clusters imply little information in t-SNE.

tfidf_df = pd.DataFrame(tsne_tfidf_sample, columns=['x', 'y'])
tfidf_df['description'] = combined_sample['item_description']
tfidf_df['tokens'] = combined_sample['tokens']
tfidf_df['category'] = combined_sample['general_cat']

source = ColumnDataSource(data=dict(
    x=tsne_tfidf_sample[:,0],
    y=tsne_tfidf_sample[:,1],
    description=list(tfidf_df['description']),
    tokens=list(combined_sample['tokens']),
    category=list(combined_sample['general_cat'])
    ))

#    tokens=list(tfidf_df['tokens']),
#    category=list(tfidf_df['category']
#x = tsne_tfidf[:,0]
#y = tsne_tfidf[:,1]

bp.output_file("tfidfclustering.html")

plot_tfidf = bp.figure(plot_width=700, plot_height=600,
                       title="tf-idf clustering of the item description",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

plot_tfidf.scatter(x = 'x', y='y', source = source, alpha=0.7)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"description": "@description", "tokens": "@tokens", "category":"@category"}
bp.show(plot_tfidf)

## K-Means Clustering

num_clusters = 30 # need to be selected wisely
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters,
                               init='k-means++',
                               n_init=1,
                               init_size=100, batch_size=100, verbose=0, max_iter=1000)

kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans.predict(vz)
kmeans_distances = kmeans.transform(vz)

## [:, ::-1] every rows, all columns in reverse order
sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(num_clusters):
    print("Cluster %d:" % i)
    aux = ''
    for j in sorted_centroids[i, :10]:
        aux += terms[j] + ' | '
    print(aux)
    print() 
    
# repeat the same steps for the sample
kmeans_sample = kmeans_model.fit(vz_sample)
kmeans_clusters_sample = kmeans.predict(vz_sample)
kmeans_distances_sample = kmeans.transform(vz_sample)

# In order to plot these clusters, first we will need to reduce the dimension
# of the distances from 30 to 2 using tsne:
tsne_kmeans_sample = tsne_model.fit_transform(kmeans_distances_sample)

#combined_sample.reset_index(drop=True, inplace=True)
kmeans_df = pd.DataFrame(tsne_kmeans_sample, columns=['x', 'y'])
kmeans_df['cluster'] = kmeans_clusters_sample
kmeans_df['description'] = combined_sample['item_description']
kmeans_df['category'] = combined_sample['general_cat']
#kmeans_df['cluster']=kmeans_df.cluster.astype(str).astype('category')

colors_list = [
    "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*cm.viridis(colors.Normalize()(kmeans_clusters_sample))
]

#cmap = cm.get_cmap('Spectral')
#norm = colors.Normalize(vmin=0.0, vmax=29.0)
#color_df = list(filter(lambda x: cmap(norm(x)), kmeans_clusters_sample))

plot_kmeans = bp.figure(plot_width=700, plot_height=600,
                        title="KMeans clustering of the description",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)


source = ColumnDataSource(data=dict(x=kmeans_df['x'], y=kmeans_df['y'],
                                    color=colors_list,
                                    description=kmeans_df['description'],
                                    category=kmeans_df['category'],
                                    cluster=kmeans_df['cluster']))

plot_kmeans.scatter(x='x', y='y', color = 'color', source=source)
hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips={"description": "@description", "category": "@category", "cluster":"@cluster" }
bp.show(plot_kmeans)

## LDA - Topic Modeling
cvectorizer = CountVectorizer(min_df=4,
                              max_features=180000,
                              tokenizer=tokenize,
                              ngram_range=(1,2))

cvz = cvectorizer.fit_transform(combined_sample['item_description'])

lda_model = LatentDirichletAllocation(n_components=20,
                                      learning_method='online',
                                      max_iter=20,
                                      random_state=42)

X_topics = lda_model.fit_transform(cvz)

n_top_words = 10
topic_summaries = []

topic_word = lda_model.components_  # get the topic words
vocab = cvectorizer.get_feature_names()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' | '.join(topic_words)))
    
# reduce dimension to 2 using tsne
tsne_lda = tsne_model.fit_transform(X_topics)

unnormalized = np.matrix(X_topics)
doc_topic = unnormalized/unnormalized.sum(axis=1)

lda_keys = []
for i, tweet in enumerate(combined_sample['item_description']):
    lda_keys += [doc_topic[i].argmax()]
    
plot_lda = bp.figure(plot_width=700, plot_height=600,
                        title="LDA clustering of the description",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

lda_df = pd.DataFrame(tsne_lda, columns=['x','y'])
lda_df['description'] = combined_sample['item_description']
lda_df['category'] = combined_sample['general_cat']
lda_df['topic'] = lda_keys
lda_df['topic'] = lda_df['topic'].map(int)

colors_list = [
    "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*cm.viridis(colors.Normalize()(lda_keys))
]

source = ColumnDataSource(data=dict(x=lda_df['x'], y=lda_df['y'],
                                    color=colors_list,
                                    description=lda_df['description'],
                                    topic=lda_df['topic'],
                                    category=lda_df['category']))

plot_lda.scatter(source=source, x='x', y='y', color='color')
hover = plot_kmeans.select(dict(type=HoverTool))
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips={"description":"@description",
                "topic":"@topic", "category":"@category"}
bp.show(plot_lda)

def prepareLDAData():
    data = {
        'vocab': vocab,
        'doc_topic_dists': doc_topic,
        'doc_lengths': list(lda_df['len_docs']),
        'term_frequency':cvectorizer.vocabulary_,
        'topic_term_dists': lda_model.components_
    } 
    return data

lda_df['len_docs'] = combined_sample['tokens'].map(len)
ldadata = prepareLDAData()
pyLDAvis.enable_notebook()
prepared_data = pyLDAvis.prepare(**ldadata)
pyLDAvis.show(prepared_data)