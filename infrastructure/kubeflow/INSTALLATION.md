

```bash
cd infrastructure/kubeflow/terraform
terraform init
terraform apply -var-file=common.tfvars
```


```bash
gcloud container clusters get-credentials deploykf --region us-central1-c --project linear-cinema-448809-v7
```


```bash
git clone -b main https://github.com/deployKF/deployKF.git ./deploykf
cd deploykf
git checkout v0.1.4
chmod +x ./argocd-plugin/install_argocd.sh
bash ./argocd-plugin/install_argocd.sh
```

```bash
nano deploykf-app-of-apps.yaml
````

Copy/paste the following content in the file:
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: deploykf-app-of-apps
  namespace: argocd
  labels:
    app.kubernetes.io/name: deploykf-app-of-apps
    app.kubernetes.io/part-of: deploykf
spec:
  project: "default"
  source:
    ## source git repo configuration
    ##  - we use the 'deploykf/deploykf' repo so we can read its 'sample-values.yaml'
    ##    file, but you may use any repo (even one with no files)
    ##
    repoURL: "https://github.com/deployKF/deployKF.git"
    targetRevision: "v0.1.4" # <-- replace with a deployKF repo tag!
    path: "."

    ## plugin configuration
    ##
    plugin:
      name: "deploykf"
      parameters:

        ## the deployKF generator version
        ##  - available versions: https://github.com/deployKF/deployKF/releases
        ##
        - name: "source_version"
          string: "0.1.4" # <-- replace with a deployKF generator version!

        ## paths to values files within the `repoURL` repository
        ##  - the values in these files are merged, with later files taking precedence
        ##  - we strongly recommend using 'sample-values.yaml' as the base of your values
        ##    so you can easily upgrade to newer versions of deployKF
        ##
        - name: "values_files"
          array:
            - "./sample-values.yaml"

        ## a string containing the contents of a values file
        ##  - this parameter allows defining values without needing to create a file in the repo
        ##  - these values are merged with higher precedence than those defined in `values_files`
        ##
        - name: "values"
          string: |
            ##
            ## This demonstrates how you might structure overrides for the 'sample-values.yaml' file.
            ## For a more comprehensive example, see the 'sample-values-overrides.yaml' in the main repo.
            ##
            ## Notes:
            ##  - YAML maps are RECURSIVELY merged across values files
            ##  - YAML lists are REPLACED in their entirety across values files
            ##  - Do NOT include empty/null sections, as this will remove ALL values from that section.
            ##    To include a section without overriding any values, set it to an empty map: `{}`
            ##

            ## --------------------------------------------------------------------------------
            ##                              deploykf-dependencies
            ## --------------------------------------------------------------------------------
            deploykf_dependencies:

              ## --------------------------------------
              ##             cert-manager
              ## --------------------------------------
              cert_manager:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

              ## --------------------------------------
              ##                 istio
              ## --------------------------------------
              istio:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

              ## --------------------------------------
              ##                kyverno
              ## --------------------------------------
              kyverno:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

            ## --------------------------------------------------------------------------------
            ##                                  deploykf-core
            ## --------------------------------------------------------------------------------
            deploykf_core:

              ## --------------------------------------
              ##             deploykf-auth
              ## --------------------------------------
              deploykf_auth:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

              ## --------------------------------------
              ##        deploykf-istio-gateway
              ## --------------------------------------
              deploykf_istio_gateway:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

              ## --------------------------------------
              ##      deploykf-profiles-generator
              ## --------------------------------------
              deploykf_profiles_generator:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

            ## --------------------------------------------------------------------------------
            ##                                   deploykf-opt
            ## --------------------------------------------------------------------------------
            deploykf_opt:

              ## --------------------------------------
              ##            deploykf-minio
              ## --------------------------------------
              deploykf_minio:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

              ## --------------------------------------
              ##            deploykf-mysql
              ## --------------------------------------
              deploykf_mysql:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

            ## --------------------------------------------------------------------------------
            ##                                  kubeflow-tools
            ## --------------------------------------------------------------------------------
            kubeflow_tools:

              ## --------------------------------------
              ##                 katib
              ## --------------------------------------
              katib:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

              ## --------------------------------------
              ##               notebooks
              ## --------------------------------------
              notebooks:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

              ## --------------------------------------
              ##               pipelines
              ## --------------------------------------
              pipelines:
                {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

  destination:
    server: "https://kubernetes.default.svc"
    namespace: "argocd"
```

```bash
kubectl apply -f ./deploykf-app-of-apps.yaml
```

Run the following command until everything works fine:
```bash
curl -fL -o "sync_argocd_apps.sh" \
  "https://raw.githubusercontent.com/deployKF/deployKF/v0.1.4/scripts/sync_argocd_apps.sh"

chmod +x ./sync_argocd_apps.sh

kubectl config current-context

bash ./sync_argocd_apps.sh
```

# Get IP address of the ingress
```bash
kubectl get svc deploykf-gateway -n deploykf-istio-gateway
```

# Set hosts
Replace `34.42.45.197` by the IP address of the ingress
```bash
34.42.45.197 deploykf.example.com
34.42.45.197 argo-server.deploykf.example.com
34.42.45.197 minio-api.deploykf.example.com
34.42.45.197 minio-console.deploykf.example.com
```

# Access the Kubeflow UI

Open a browser and go to `https://deploykf.example.com:8443`