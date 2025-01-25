import kfp.dsl as dsl_v1
from kfp.v2 import dsl
from kfp import compiler
from pipelines.solution.components import (
    load_and_split_data,
    preprocess_data,
    evaluate_model,
    tf_minio_to_gcs,
    tf_model_gcs_to_minio
)

from kfp.v2.dsl.experimental import component

from kubernetes.client import V1EnvVar, V1EnvVarSource, V1SecretVolumeSource, V1Volume, V1VolumeMount
@component(
    packages_to_install=["kubeflow-training==1.6.0"],
    base_image="python:3.10",
)
def train_model_dist(
    job_name: str,
    namespace: str,
    learning_rate: float,
    num_epoch: int,
    train_dataset_path: str,
    model_artifact_path: str,
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
        V1EnvVar,
        V1Volume,
        V1VolumeMount,
        V1SecretVolumeSource,
    )

    volume = V1Volume(
        name="google-cloud-key", secret=V1SecretVolumeSource(secret_name="gcp-sa")
    )

    volume_mount = V1VolumeMount(
        name="google-cloud-key", mount_path="/var/secrets/google", read_only=True
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
                                        "--train_dataset_path",
                                        train_dataset_path,
                                        "--model_artifact_path",
                                        model_artifact_path,
                                    ],
                                    resources=V1ResourceRequirements(
                                        limits={"memory": "1Gi", "cpu": "0.75"}
                                    ),
                                    image_pull_policy="Always",
                                    volume_mounts=[volume_mount],
                                    env=[
                                        V1EnvVar(
                                            name="GOOGLE_APPLICATION_CREDENTIALS",
                                            value="/var/secrets/google/google-key.json",
                                        )
                                    ],
                                )
                            ],
                            volumes=[volume],
                        ),
                    ),
                )
            },
        ),
    )

    training_client = TrainingClient()
    training_client.create_tfjob(tfjob=tfjob, namespace=namespace)

    training_client.wait_for_job_conditions(name=job_name, namespace=namespace)

    print(training_client.get_job_logs(name=job_name, namespace=namespace))


@dsl.pipeline(name="mnist-training-pipeline")
def training_pipeline(
    gcs_bucket: str,
    namespace: str,
    training_job_name: str,
    training_job_image: str,
    learning_rate: float,
    num_epoch: int,
):
    load_and_split_data_op = load_and_split_data()

    preprocess_train_data_op = preprocess_data(
        dataset=load_and_split_data_op.outputs["train_dataset"],
        size=load_and_split_data_op.outputs["train_size"],
    )

    preprocess_test_data_op = preprocess_data(
        dataset=load_and_split_data_op.outputs["test_dataset"],
        size=load_and_split_data_op.outputs["test_size"],
    )

    tf_minio_to_gcs_op = tf_minio_to_gcs(
        tf_dataset=preprocess_train_data_op.outputs["preprocessed_dataset"],
        gcs_bucket=gcs_bucket,
    ).set_caching_options(False)#.add_env_variable(V1EnvVar(
    #     name='GOOGLE_APPLICATION_CREDENTIALS',
    #     value='/secrets/google-key.json'
        
    # ))


    # secret_volume = V1Volume(
    #     name='google-credentials',
    #     secret=V1SecretVolumeSource(
    #         secret_name='gcp-sa'
    #     )
    # )
    # secret_volume_mount = V1VolumeMount(
    #     name='google-credentials',
    #     mount_path='/secrets',
    #     read_only=True
    # )
    
    # tf_minio_to_gcs_op.add_volume(secret_volume)
    # tf_minio_to_gcs_op.add_volume_mount(secret_volume_mount)

    # train_model_op = train_model_dist(
    #     job_name=training_job_name,
    #     namespace=namespace,
    #     learning_rate=str(learning_rate),
    #     num_epoch=str(num_epoch),
    #     train_dataset_path=tf_minio_to_gcs_op.outputs["dataset_gcs_uri"],
    #     model_artifact_path=tf_minio_to_gcs_op.outputs["model_gcs_uri"],
    #     image=training_job_image,
    # )

    # tf_model_gcs_to_minio_op = tf_model_gcs_to_minio(
    #     tf_minio_to_gcs_op.outputs["model_gcs_uri"]
    # )

    # tf_model_gcs_to_minio_op.after(train_model_op)

    # evaluate_model(
    #     test_dataset=preprocess_test_data_op.outputs["preprocessed_dataset"],
    #     model_artifact=tf_model_gcs_to_minio_op.outputs["model_artifact"],
    # )


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
        "--training_job_image", type=str, default="tensorflow/tensorflow:2.13.0rc2"
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
            "gcs_bucket": args.gcs_bucket,
            "namespace": args.namespace,
            "training_job_name": args.training_job_name,
            "learning_rate": args.learning_rate,
            "num_epoch": args.num_epoch,
            "training_job_image": args.training_job_image,
        },
        namespace=args.namespace,
    )