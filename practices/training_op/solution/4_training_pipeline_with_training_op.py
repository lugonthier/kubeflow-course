import kfp.dsl as dsl_v1
from kfp.v2 import dsl
from kfp import compiler

from kfp.v2.dsl.experimental import component

@component(
    packages_to_install=["kubeflow-training==1.6.0"],
    base_image="python:3.10",
)
def train_model_dist(
    job_name: str,
    namespace: str,
    learning_rate: float,
    num_epoch: int,
    image: str,
):
    from kubeflow.training.constants import constants
    from kubeflow.training import (
        TrainingClient,
        V1ReplicaSpec,
        V1RunPolicy,
        KubeflowOrgV1TFJob,
        KubeflowOrgV1TFJobSpec,
    )

    from kubernetes.client import (
        V1PodTemplateSpec,
        V1ObjectMeta,
        V1PodSpec,
        V1Container,
        V1ResourceRequirements,
    )


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
                                    image=image,
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

    training_client = TrainingClient()
    training_client.create_tfjob(tfjob=tfjob, namespace=namespace)

    training_client.wait_for_job_conditions(name=job_name, namespace=namespace)

    training_client.get_job_logs(name=job_name, namespace=namespace)


@dsl.pipeline(name="mnist-training-pipeline")
def training_pipeline(
    namespace: str,
    training_job_name: str,
    training_job_image: str,
    learning_rate: float,
    num_epoch: int,
):

    train_model_op = train_model_dist(
        job_name=training_job_name,
        namespace=namespace,
        learning_rate=str(learning_rate),
        num_epoch=str(num_epoch),
        image=training_job_image,
    )


if __name__ == "__main__":
    import argparse
    from pipelines.kfp_client import KFPClientManager

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--gcs_bucket", type=str)
    parser.add_argument("--namespace", type=str, default="team-1")
    parser.add_argument("--training_job_name", type=str)
    parser.add_argument(
        "--training_job_image", type=str, default="tensorflow/tensorflow:2.15.0"
    )
    args = parser.parse_args()

    kfp_client_manager = KFPClientManager(
        api_url="https://deploykf.example.com:8443/pipeline",
        skip_tls_verify=True,
        dex_username="admin@example.com",
        dex_password="admin",
        dex_auth_type="local",
    )

    kfp_client = kfp_client_manager.create_kfp_client()

    compiler.Compiler(mode=dsl_v1.PipelineExecutionMode.V2_COMPATIBLE).compile(
        pipeline_func=training_pipeline, package_path="mnist_training_pipeline_with_training_op.yaml"
    )

    kfp_client.create_run_from_pipeline_package(
        pipeline_file="mnist_training_pipeline_with_training_op.yaml",
        arguments={
            "namespace": args.namespace,
            "training_job_name": args.training_job_name,
            "learning_rate": args.learning_rate,
            "num_epoch": args.num_epoch,
            "training_job_image": args.training_job_image,
        },
        namespace=args.namespace,
    )