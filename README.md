# Deloitte Machine Learning Challenge 2021
Deloitte Presents Machine Learning Challenge: Predict Loan Defaulters in association with [Machine Hack](https://machinehack.com/hackathon/deloitte_presents_machine_learning_challenge_predict_loan_defaulters/overview)
## Overview
* Banks run into losses when a customer doesn't pay their loans on time. Because of this, every year, banks have losses in crores, and this also impacts the country's economic growth to a large extent. In this hackathon, we look at various attributes such as funded amount, location, loan, balance, etc., to predict if a person will be a loan defaulter or not. 

* To solve this problem, MachineHack has created a training dataset of 67,463 rows and 35 columns and a testing dataset of 28,913 rows and 34 columns. The hackathon demands a few pre-requisite skills like big dataset, underfitting vs overfitting, and the ability to optimise “log_loss” to generalise well on unseen data. 
## Data Description

| Feature             
| ----------------------- 
| Customer_ID             
 ID                            
 Loan Amount                  
 Funded Amount                 
 Funded Amount Investor       
 Term                          
 Batch Enrolled               
 Interest Rate               
 Grade                         
 Sub Grade                    
 Employment Duration          
 Home Ownership                
 Verification Status           
 Payment Plan                
 Loan Title                   
 Debit to Income              
 Delinquency - two years      
 Inquires - six months         
 Open Account                  
 Public Record                 
 Revolving Balance             
 Revolving Utilities         
 Total Accounts                
 Initial List Status           
 Total Received Interest       
 Total Received Late Fee       
 Recoveries                    
 Collection Recovery Fee       
 Collection 12 months Medical  
 Application Type             
 Last week Pay                
 Accounts Delinquent             
 Total Collection Amount       
 Total Current Balance         
 Total Revolving Credit Limit  
 Loan Status      
 
 ## Skills Practiced
 * Visualization
 * Feature Engineering- different dncoding techniques
 * Multiple models-hyperparameter tuning
 * Methods to maximize the metric of evaluation
 * Stacking 
 
 from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Preprocess the text
sentences = [
    "The cat sat on the mat",
    "The dog sat on the rug",
    "The cat slept on the mat",
    "The dog slept on the rug",
    "The cat and dog were friends",
    "The cat and dog were enemies"
]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_sentences = [tokenizer.tokenize(sent) for sent in sentences]
cleaned_sentences = []
for tokens in tokenized_sentences:
    cleaned_tokens = [token for token in tokens if token.isalpha()]
    cleaned_sentences.append(" ".join(cleaned_tokens))

# Step 2: Embed the sentences
model = AutoModel.from_pretrained("bert-base-uncased")
embeddings = []
for sent in cleaned_sentences:
    input_ids = tokenizer.encode(sent, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        output = model(input_ids)[0][:, 0, :].numpy()
    embeddings.append(output)
embeddings = np.vstack(embeddings)

# Step 3: Compute similarities
similarity_matrix = cosine_similarity(embeddings)

# Step 4: Cluster the sentences
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(similarity_matrix)

# Step 5: Evaluate the clusters
cluster_labels = kmeans.labels_
print(cluster_labels) # should output [0 0 0 0 1 1]

# Step 6: Map the contexts
context_sentences = [[] for _ in range(num_clusters)]
for i, label in enumerate(cluster_labels):
    context_sentences[label].append(sentences[i])
print(context_sentences) # should output [['The cat sat on the mat', 'The dog sat on the rug', 'The cat slept on the mat', 'The dog slept on the rug'], ['The cat and dog were friends', 'The cat and dog were enemies']]

