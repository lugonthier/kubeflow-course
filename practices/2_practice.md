# Training Operator 1: K8S manifest

Concevez un manifest pour entraîner votre modèle Tensorflow sur le jeu de données MNIST. Le fichier [sample_training.yaml](training_operator/sample_training.yaml) est un bon example pour commencer.

Pour cela vous aurez besoin de construire une image Docker avec les dépendances nécessaires pour executer votre script.
Un example avec Tensorflow:
```Dockerfile
FROM tensorflow/tensorflow:2.15.0

COPY train.py .

CMD ["python", "./train.py"]
```

Pour pousser votre image vers Google Artifact Registry:

```bash
docker build -t <DOCKER_IMAGE_URI> .
````


```bash
docker push <DOCKER_IMAGE_URI>
```

# Training Operator 2: SDK with API Objects

Concevez un script utilisant le SDK avec les objets d'API afin d'entraîner votre modèle sur le jeu de données  MNIST.

Pour plus d'information sur comment déclarer les objets, suivez la documentation suivante: [KubeflowOrgV1TFJob](https://github.com/kubeflow/training-operator/blob/v1.6-branch/sdk/python/docs/KubeflowOrgV1TFJob.md)


Un exemple d'utilisation peut être trouvé ici: [test_e2e_tfjob.py](https://github.com/kubeflow/training-operator/blob/v1.6-branch/sdk/python/test/e2e/test_e2e_tfjob.py#L93C1-L144C5)

# Training Operator 3: SDK with high level interface

Concevez un script utilisant la méthode `create_tfjob_from_func` du `TrainingClient` pour entraîner votre modèle.



# Training Operator 4: Integrate the part 2 into an ML Pipeline


Utiliez le script développé dans `Training Operator 2` et intégrez le dans une pipeline Kubeflow. Puis executez la pipeline.



