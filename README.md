Purpose of model:  
Text classification

Structure of model:  
A CNN with two tri-gram filters, two four-gram filters and two five-gram filters. Max-pooled results are then fed into a softmax layer for classification.

Management of files:
* **data_helpers.py** holds utility functions regarding handling of data.
* **text_cnn.py** holds the structure of the neural network as a class.
* **Train.py** implements the training of the network.

Dataset:  
English movie review text dataset with negative and positive polarities. The dataset contains 10662 documents with maximum length document length of 56.
