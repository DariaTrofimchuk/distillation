# distillation
Distillation of the Russian News Bert Model
This repository contains a script used to build and train the Distilled model for Russian News Clustering. 
It is a supporting material for the Evaluation 2022 paper 'Distilled Model for Russian News Clustering: much lighter and faster, still accurate'.

Our work continues the research started in 2020 within the Telegram Data Clustering contest and developed in the Dialogue Evaluation 2021 task on Russian news clustering. 
Using a BERT-based clustering model as a teacher we tested various student networks based on different architectures (RNN, FFN, convolutional and Transformer-based networks) in order to get a faster lightweight analogue that is more likely to be deployed in real products. 

We tried two distillation strategies: the first one combined an original loss function from the initial model with a distillation objective, for the second one we used only a specific distillation loss. This approach turned out to be more successful. 

Both strategies are presented in this notebook. We reproduce the training process of our best tested model, as well as other distillation experiments.  
