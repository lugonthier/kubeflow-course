# 0 - Installation des dépendances

Pour cela il suffit d'installer les librairies du fichier [requirements.txt](requirements.txt) dans un nouvel environnement python.



# 1 - Pipeline d'entraînement

Dans cette première pratique vous devez créer une pipeline d'entraînement ayant pour but de produire un modèle pour le jeu de données MNIST. Veilliez à créer des composants logiques et ayant une seule fonction.


Pour démarrer avec Kubeflow Pipelines vous pouvez vous inspirez de la pipeline dans le fichier [sample_pipeline.py](pipelines/sample_pipeline.py). Pour executer cette pipeline il vous suffit simplement d'ajouter votre username et password dans le `KFPClientManager`.

Le but n'étant pas de s'exercer au machine learning mais à Kubeflow voici un example que vous pouvez reprendre utilisant Tensorflow: https://www.tensorflow.org/datasets/keras_example?hl=fr