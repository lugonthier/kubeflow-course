import argparse
from kubernetes.client import V1ObjectMeta
from kubeflow.katib import (
    V1beta1TrialTemplate,
    KatibClient,
    V1beta1ExperimentSpec,
    V1beta1Experiment,
    V1beta1AlgorithmSpec,
    V1beta1FeasibleSpace,
    V1beta1ObjectiveSpec,
    V1beta1ParameterSpec,
    V1beta1TrialParameterSpec,
    V1beta1EarlyStoppingSetting,
    V1beta1EarlyStoppingSpec
)


def create_katib_experiment(
    experiment_name: str,
    namespace: str,
    base_image: str,
    train_dataset_path: str,
    model_artifact_path: str,
) -> V1beta1Experiment:
    objective_spec = V1beta1ObjectiveSpec(
        type="maximize",
        objective_metric_name="accuracy",
    )

    parameters = [
        V1beta1ParameterSpec(
            name="lr",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(min="0.001", max="0.1"),
        ),
    ]

    algorithm_spec = V1beta1AlgorithmSpec(algorithm_name="random")

    early_stopping = V1beta1EarlyStoppingSpec(
        algorithm_name="medianstop",
        algorithm_settings=[
            V1beta1EarlyStoppingSetting(name="min_trials_required", value="5")
        ],
    )

    trial_template = V1beta1TrialTemplate(
        primary_container_name="training-container",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="LearningRate",
                description="Optimizer Learning Rate",
                reference="lr",
            )
        ],
        
        trial_spec={
            "apiVersion": "batch/v1",
            "kind": "Job",
            "spec": {
                "template": {
                    "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "training-container",
                                "image": base_image,
                                "imagePullPolicy": "Always",
                                "command": [
                                    "python",
                                    "./train.py",
                                    "--lr=${trialParameters.LearningRate}",
                                    "--num_epoch=10",
                                    f"--train_dataset_path={train_dataset_path}",
                                    f"--model_artifact_path={model_artifact_path}",
                                    '--is_dist="False"',
                                    '--num_workers="1"',
                                ],
                                "env": [
                                    {
                                        "name": "GOOGLE_APPLICATION_CREDENTIALS",
                                        "value": "/var/secrets/google/google-key.json",
                                    }
                                ],
                                "volumeMounts": [
                                    {
                                        "name": "google-cloud-key",
                                        "mountPath": "/var/secrets/google",
                                    }
                                ],
                            }
                        ],
                        "restartPolicy": "Never",
                        "volumes": [
                            {
                                "name": "google-cloud-key",
                                "secret": {"secretName": "gcp-sa"},
                            }
                        ],
                    },
                }
            },
        },
    )

    experiment_spec = V1beta1ExperimentSpec(
        max_trial_count=3,
        parallel_trial_count=1,
        max_failed_trial_count=1,
        objective=objective_spec,
        parameters=parameters,
        trial_template=trial_template,
        algorithm=algorithm_spec,
        early_stopping=early_stopping,
    )

    experiment = V1beta1Experiment(
        api_version="kubeflow.org/v1beta1",
        kind="Experiment",
        spec=experiment_spec,
        metadata=V1ObjectMeta(name=experiment_name, namespace=namespace),
    )

    return experiment


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, help="")
    parser.add_argument("--experiment_name", type=str, help="")
    parser.add_argument("--base_image", type=str, help="")
    parser.add_argument("--train_dataset_path", type=str, help="")
    parser.add_argument("--model_artifact_path", type=str, help="")

    args = parser.parse_args()

    katib_client = KatibClient()

    experiment = create_katib_experiment(
        args.experiment_name,
        args.namespace,
        base_image=args.base_image,
        train_dataset_path=args.train_dataset_path,
        model_artifact_path=args.model_artifact_path,
    )
    katib_client.create_experiment(experiment=experiment, namespace=args.namespace)

    katib_client.wait_for_experiment_condition(
        name=args.experiment_name, namespace=args.namespace
    )

    opt = katib_client.get_optimal_hyperparameters(
        name=args.experiment_name, namespace=args.namespace
    )