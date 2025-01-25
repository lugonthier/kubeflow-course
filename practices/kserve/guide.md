

custom model: https://github.com/kserve/kserve/tree/master/python/custom_model


# 1 - Deploy KServe locally

Follow [Kserve getting started page](https://kserve.github.io/website/0.11/get_started/) or follow indication below:

Prerequisites:
- Docker installed ([installation guide](https://docs.docker.com/engine/install/))
- Kind installed ([installation guide](https://kind.sigs.k8s.io/docs/user/quick-start))
- Kubernetes CLI (kubectl) installed ([installation guide](https://kubernetes.io/docs/tasks/tools/))


### 1.1 - Create the cluster
```bash
kind create cluster
```

Use the newly created context:
```bash
kubectl config use-context kind-kind
```
### 1.2 - Install KServe
```bash
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.11/hack/quick_install.sh" | bash
```

### 1.3 - Deploy a sample model to ensure KServe is working (optional)
Follow [the official documentation](https://kserve.github.io/website/0.11/get_started/first_isvc/) or instructions below:

#### 1.3.1 - Create a namespace

```bash
kubectl create namespace kserve-test
```

#### 1.3.2 - Create an InferenceService using a k8s manifest 

```bash
kubectl apply -n kserve-test -f - <<EOF
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
EOF
```

Check it is deployed:
```bash
kubectl get inferenceservices sklearn-iris -n kserve-test
```

#### 1.3.3 - Test the newly created endpoint

In a terminal run the two commands below:
```bash
INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
```

```bash
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80
```

In another terminal run the four commands below:

```bash
export INGRESS_HOST=localhost
export INGRESS_PORT=8080
```

```bash
cat <<EOF > "./iris-input.json"
{
  "instances": [
    [6.8,  2.8,  4.8,  1.4],
    [6.0,  3.4,  4.5,  1.6]
  ]
}
EOF
```
```bash
SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -n kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./iris-input.json
```

If you see the predictions then KServe was successfully installed and is working.


# 2. Deploy a model on KServe

KServe defines model serving environments using two CRDs ([Custom Resources Definition](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)), **ServingRuntimes** and **ClusterServingRuntimes**.

Both are very similar, to know the only differences between the two, [read the official documentation](https://kserve.github.io/website/0.11/modelserving/servingruntimes/).

## 2.1 - Deploy a model using a standard KServe predictor
KServe provide out-of-the-box runtimes to quickly deploy common model formats like TensorFlow, PyTorch, Scikit-Learn ([see full list](https://kserve.github.io/website/0.11/modelserving/servingruntimes/)):

| Name                       | Supported Model Formats |
| -------------------------- | ----------------------- |
| kserve-tensorflow-serving  | TensorFlow              |
| kserve-torchserve          | PyTorch                 |
| kserve-sklearnserver       | SKLearn                 |

When defining the predictor in an InferenceService, we can explicitly specify the name of the runtime:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: example-sklearn-isvc
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: s3://bucket/sklearn/mnist.joblib
      runtime: kserve-mlserver # <--- Explicitly specify runtime
```

However, we can also the let KServe find a runtime that meets our needs. KServe will look for a runtime that support the specified model format:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: example-sklearn-isvc
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn # <--- Use the model format look for a runtime.
      storageUri: s3://bucket/sklearn/mnist.joblib
```

After writing the InferenceService you can just apply it:

```bash

kubectl apply -f inference-service.yaml

```

## 2.2 - Deploy a model using a Custom Predictor

Using prebuilt runtime is fine for quick deployment and simple models. However, when complexity increases or specific optimizations are required, custom runtimes offer more flexibility and control over the deployment environment.


### 2.2.1 - Implement Custom Model using KServe API
First, we need to write the service which is defined as a `Model` with 2 required methods, `load` and `predict`, and 2 optional methods, `preprocess` and `postprocess`. The method names talk for themselves.

Below an example:
```python
from kserve import InferRequest, InferResponse, InferOutput, Model, ModelServer

# Step 1: Define model logic.
class SampleModel(Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.load()

    def load(self):
      # Load logic
      self.ready = True
    
    def predict(self, payload: InferRequest, headers: Dict[str, str] = None) -> InferResponse:
        # Process Inputs
        return InferResponse(...)



# Step 2: Start the server
if __name__ == "__main__":
  model = SampleModel("example-model")
  ModelServer().start([model])
```

### 2.2.2 - Build Serving Environment

Simply need to build a docker image.
Example below:
```Dockerfile
FROM python:3.10-slim-bullseye

COPY requirements.txt requirements.txt
COPY model.py model.py

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "-m", "model"]

```

Then build and push to whatever is you image registry:

```bash
docker build -t <IMAGE_URI> .
```

Let's push it:

```bash
docker push <IMAGE_URI>
```

### 2.2.3 - Inference Service Resource

The k8s cluster will need rights to pull the image from the image registry. To allow it, we need to create a kubernetes secret (in GCP using a service account key for example):


```bash
kubectl create secret docker-registry sa-key \
    --docker-server=<LOCATION>-docker.pkg.dev/<PROJECT-ID>/<REPOSITORY> \
    --docker-username=_json_key \
    --docker-password="$(cat sa-key.json)"
```

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: custom-model
spec:
  predictor:
    imagePullSecrets:
      - name: sa-key # <--- Don't forget to specify secret.
    containers:
      - name: kserve-container
        image: <IMAGE_URI>
```

Then apply the Inference Service:
```bash
kubectl apply -f inference-service.yaml
```

## 2.3 - Deploy a model using the KServe SDK

Sometimes (most of the time ?), you want to deploy a model through a ML pipeline. However, a k8s manifest as part of a pipeline might not be a good practice. In such a case, the KServe SDK is what we need. 

The same way you can do it for other Kubeflow components (like `Training Operator` and `Katib`), you can specify a yaml manifest in python using the KServe SDK.

Example below:

```python
from kubernetes import client
from kserve import (V1beta1InferenceService, V1beta1InferenceServiceSpec,
                    V1beta1PredictorSpec, constants, KServeClient)

default_model_spec = V1beta1InferenceServiceSpec(
    predictor=V1beta1PredictorSpec(
        image_pull_secrets=[
            client.V1LocalObjectReference(
                name='sa-key'
            )
            ],
        containers=[
            client.V1Container(
            name="kserve-container",
            image="<IMAGE_URI>"
        )]
    )
)

isvc = V1beta1InferenceService(
    api_version=constants.KSERVE_V1BETA1,
    kind=constants.KSERVE_KIND,
    metadata=client.V1ObjectMeta(name="custom-model"),
    spec=default_model_spec
)
```

After defining the Inference Service we can use the KServe python client to create, update or delete it.
Below a create example:
```python
kserve = KServeClient()
kserve.create(isvc, # <--- The `V1beta1InferenceService` previously instantiated.
              watch=True # <--- Set `watch` to True to wait until the Inference Service is ready.
              )
```