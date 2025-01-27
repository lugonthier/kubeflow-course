from kubeflow.training.constants import constants
from kubeflow.training import (
    TrainingClient,
    V1ReplicaSpec,
    V1RunPolicy,
    KubeflowOrgV1TFJob,
    KubeflowOrgV1TFJobSpec
)


from kubernetes.client import (
    V1PodTemplateSpec,
    V1ObjectMeta,
    V1PodSpec,
    V1Container,
    V1ResourceRequirements,
)

def create_tf_job_spec(
    job_name: str,
    base_image: str,
    namespace: str,
    learning_rate: float,
    num_epoch: int,
):


    tfjob = KubeflowOrgV1TFJob(
        api_version="kubeflow.org/v1",
        kind=constants.TFJOB_KIND,
        metadata=V1ObjectMeta(name=job_name, namespace=namespace),
        spec=KubeflowOrgV1TFJobSpec(
            run_policy=V1RunPolicy(
                clean_pod_policy=None,
                scheduling_policy=None,
            ),
            tf_replica_specs={
                "Worker": V1ReplicaSpec(
                    replicas=1,
                    restart_policy="Never",
                    template=V1PodTemplateSpec(
                        metadata=V1ObjectMeta(
                            name=job_name,
                            namespace=namespace,
                            annotations={"sidecar.istio.io/inject": "false"},
                        ),
                        spec=V1PodSpec(
                            containers=[
                                V1Container(
                                    name="tensorflow",
                                    image=base_image,
                                    command=[
                                        "python",
                                        "./train.py",
                                        "--lr",
                                        str(learning_rate),
                                        "--num_epoch",
                                        str(num_epoch),
                                    ],
                                    resources=V1ResourceRequirements(
                                        limits={"memory": "1Gi", "cpu": "0.75"}
                                    ),
                                    image_pull_policy="Always",
                                    
                                )
                            ],
                        ),
                    ),
                )
            },
        ),
    )
    return tfjob


if __name__ == "__main__":
    import argparse

    TRAINING_CLIENT = TrainingClient()

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--namespace", type=str, default="team-1")
    parser.add_argument("--training_job_name", type=str)
    parser.add_argument(
        "--training_job_image", type=str, default="tensorflow/tensorflow:2.13.0rc2"
    )
    args = parser.parse_args()

    tfjob = create_tf_job_spec(
        job_name=args.training_job_name,
        namespace=args.namespace,
        base_image=args.training_job_image,
        learning_rate=args.learning_rate,
        num_epoch=args.num_epoch,
    )

    TRAINING_CLIENT.create_tfjob(tfjob=tfjob, namespace=args.namespace)

    TRAINING_CLIENT.wait_for_job_conditions(name=args.training_job_name, namespace=args.namespace)

    print(TRAINING_CLIENT.get_job_logs(name=args.training_job_name, namespace=args.namespace))