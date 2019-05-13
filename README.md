# Yelp Review Sentiment Prediction using NLP (Glove)
This repo demonstrate a sentiment prediction based on Yelp reviews using natural language processing based on Glove word embeddings in Keras.

##  How to Use:
    Clone in local directory
        git clone git@github.com:mmortazavi/yelp_review_sentiment_prediction_nlp.git
    Download `glove.6B.100d.txt` Glove Embedding from this link [https://www.kaggle.com/terenceliu4444/glove6b100dtxt], and Put it under the embedding directory
        copy glove.6B.100d.txt embedding
    Navigate into yelp_review_sentiment_prediction_nlp directory, and run:
        python setup.py
      
##  Results after 20 Epochs on 1 Millions Data Points:
    
    Epoch 1/20
    80000/80000 [==============================] - 217s 3ms/step - loss: 0.4156 - acc: 0.8058 - val_loss: 0.3105 - val_acc: 0.8642
    Epoch 2/20
    80000/80000 [==============================] - 201s 3ms/step - loss: 0.3136 - acc: 0.8650 - val_loss: 0.2947 - val_acc: 0.8739
    Epoch 3/20
    80000/80000 [==============================] - 192s 2ms/step - loss: 0.2888 - acc: 0.8768 - val_loss: 0.2743 - val_acc: 0.8834
    Epoch 4/20
    80000/80000 [==============================] - 193s 2ms/step - loss: 0.2726 - acc: 0.8845 - val_loss: 0.2619 - val_acc: 0.8883
    Epoch 5/20
    80000/80000 [==============================] - 174s 2ms/step - loss: 0.2605 - acc: 0.8907 - val_loss: 0.2559 - val_acc: 0.8930
    Epoch 6/20
    80000/80000 [==============================] - 172s 2ms/step - loss: 0.2488 - acc: 0.8964 - val_loss: 0.2637 - val_acc: 0.8901
    Epoch 7/20
    80000/80000 [==============================] - 172s 2ms/step - loss: 0.2408 - acc: 0.9004 - val_loss: 0.2637 - val_acc: 0.8887
    Epoch 8/20
    80000/80000 [==============================] - 171s 2ms/step - loss: 0.2327 - acc: 0.9036 - val_loss: 0.2615 - val_acc: 0.8901
    Epoch 9/20
    80000/80000 [==============================] - 203s 3ms/step - loss: 0.2259 - acc: 0.9074 - val_loss: 0.2477 - val_acc: 0.8953
    Epoch 10/20
    80000/80000 [==============================] - 178s 2ms/step - loss: 0.2169 - acc: 0.9116 - val_loss: 0.2484 - val_acc: 0.8966
    Epoch 11/20
    80000/80000 [==============================] - 183s 2ms/step - loss: 0.2123 - acc: 0.9141 - val_loss: 0.2486 - val_acc: 0.8981
    Epoch 12/20
    80000/80000 [==============================] - 183s 2ms/step - loss: 0.2063 - acc: 0.9163 - val_loss: 0.2553 - val_acc: 0.8962
    Epoch 13/20
    80000/80000 [==============================] - 170s 2ms/step - loss: 0.2020 - acc: 0.9185 - val_loss: 0.2493 - val_acc: 0.8971
    Epoch 14/20
    80000/80000 [==============================] - 170s 2ms/step - loss: 0.1951 - acc: 0.9223 - val_loss: 0.2522 - val_acc: 0.8960
    Epoch 15/20
    80000/80000 [==============================] - 164s 2ms/step - loss: 0.1913 - acc: 0.9225 - val_loss: 0.2592 - val_acc: 0.8956
    Epoch 16/20
    80000/80000 [==============================] - 162s 2ms/step - loss: 0.1856 - acc: 0.9251 - val_loss: 0.2645 - val_acc: 0.8913
    Epoch 17/20
    80000/80000 [==============================] - 171s 2ms/step - loss: 0.1818 - acc: 0.9280 - val_loss: 0.2587 - val_acc: 0.8949
    Epoch 18/20
    80000/80000 [==============================] - 200s 3ms/step - loss: 0.1764 - acc: 0.9301 - val_loss: 0.2581 - val_acc: 0.8951
    Epoch 19/20
    80000/80000 [==============================] - 184s 2ms/step - loss: 0.1732 - acc: 0.9304 - val_loss: 0.2670 - val_acc: 0.8915
    Epoch 20/20
    80000/80000 [==============================] - 203s 3ms/step - loss: 0.1689 - acc: 0.9327 - val_loss: 0.2754 - val_acc: 0.8923
    Training Complete in 3666.44 s.
    ----------------------------------------------------------------------------------------------------

    Score: 89.22

    Classification Report:
                   precision    recall  f1-score   support

        Negative       0.85      0.82      0.84      6706
        Positive       0.91      0.93      0.92     13294

       micro avg       0.89      0.89      0.89     20000
       macro avg       0.88      0.88      0.88     20000
    weighted avg       0.89      0.89      0.89     20000

    ----------------------------------------------------------------------------------------------------
## The Repo Contains:
    - data dir: a small fraction of yelp reviews in parquet format
    - notebook dir: 
        - 'EDA' exploratory data analysis  on Yelp reviews
        - 'NLP-Glove_SentimentAnalysis' demonstrate a run of the model and results
    - resutls dir: figures i.e. model training history
    - setup.py, main file for running the project. 
      important: please note some variables like DATA_PATH, DATA_FILE, RESULT_PATH, EMBEDDING_PATH, EMBEDDING_VEC, EMBEDDING_DIM need to be modified.
    - yelp dir: list of all python modules
    
##  Easy Approach:Run the Docker Container
    - pull the image:
    docker pull majidmortazavi/yelp_review_sentiment_prediction_nlp
    - run the image:
    docker run -t yelp_review_sentiment_prediction_nlp
