# Katib 1: K8S manifest

Concevez un manifest pour effectuer une recherche d'hyperparamètre pour votre modèle Tensorflow sur le jeu de données MNIST avec la configuration suivante:

- maximum trial count = 5
- maximum parallel trial = 1

Comme pour le Training Operator, vous aurez besoin ici de spécifier une image Docker.

Le fichier [sample_tuning.yaml](training_operator/sample_tuning.yaml) est un bon example pour commencer.

Après l'execution terminée, vous pouvez observer les résultats dans l'onglet Katib.

# Katib 2: SDK with API Objects

Concevez un script utilisant le SDK avec les objets d'API afin d'optimiser votre modèle sur le jeu de données MNIST.

Pour plus d'information sur comment déclarer les objets, suivez la documentation suivante: [V1beta1Experiment](https://github.com/kubeflow/katib/blob/release-0.15/sdk/python/v1beta1/docs/V1beta1Experiment.md)


# Katib 3: SDK with high level interface

Concevez un script utilisant la méthode `tune` du `KatibClient` pour optimiser cette fois-ci le modèle de classification des iris créé dans la pratique 2 sur KFP.

Vous pouvez trouver un example de comment l'utiliser ici: [tune-train-from-func.ipynb](https://github.com/kubeflow/katib/blob/master/examples/v1beta1/sdk/tune-train-from-func.ipynb)

# Katib 4: Integrate part 2 into KFP.

Dans le chapitre 2 sur KFP vous avez conçu une pipeline d'entraînement. Intégrez la partie `Katib 3` dans votre pipeline pour optimiser le modèle.