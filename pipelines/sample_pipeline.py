import kfp.dsl as dsl_v1
from kfp.v2 import dsl
from kfp import compiler
from kfp.v2.dsl.experimental import component, Output, Dataset, Model, Input

@dsl.component(packages_to_install=["pandas", "scikit-learn"])
def load_data(dataset: Output[Dataset]):
    from sklearn.datasets import load_iris
    import pandas as pd


    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    
    
    iris_df.to_csv(dataset.path, index=False)


@dsl.component(packages_to_install=["pandas", "scikit-learn"])
def preprocess_data(dataset: Input[Dataset], preprocessed_dataset: Output[Dataset]):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(dataset.path)
    
    preprocessed_data = StandardScaler().fit_transform(df)
    
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=df.columns)
    
    
    preprocessed_df.to_csv(preprocessed_dataset.path)
    
    
@dsl.pipeline(name="sample-pipeline")
def sample_pipeline():
    
    load_data_op = load_data()
    preprocess_data_op = preprocess_data(dataset=load_data_op.outputs['dataset'])
    
    
if __name__=="__main__":    
    from pipelines.kfp_client import KFPClientManager

    dex_username="user1@example.com"
    dex_password="user1"
    
    kfp_client_manager = KFPClientManager(
        api_url="https://deploykf.example.com:8443/pipeline",
        skip_tls_verify=True,

        dex_username=dex_username,
        dex_password=dex_password,

        dex_auth_type="local",
    )

    kfp_client = kfp_client_manager.create_kfp_client()

    compiler.Compiler(mode=dsl_v1.PipelineExecutionMode.V2_COMPATIBLE).compile(
        pipeline_func=sample_pipeline,
        package_path=f'sample_pipeline_{dex_username}.yaml'
    )

    kfp_client.create_run_from_pipeline_package(
        pipeline_file=f'sample_pipeline_{dex_username}.yaml',
        arguments={},
        namespace='team-1',
        enable_caching=False
    )