import kfp.dsl as dsl_v1
from kfp.v2 import dsl
from kfp import compiler
from kfp.v2.dsl.experimental import component, Output, Dataset, Model, Input

@dsl.component(packages_to_install=["pandas"])
def create_dataset(dataset: Output[Dataset]):
    import pandas as pd

    data = {
        'number': range(1, 101),
        'square': [x**2 for x in range(1, 101)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(dataset.path, index=False)


@dsl.component(packages_to_install=["pandas"])
def operations(dataset: Input[Dataset], preprocessed_dataset: Output[Dataset]):
    import pandas as pd
    
    df = pd.read_csv(dataset.path)
    
    df['cube'] = df['number'] ** 3
    df['normalized'] = df['number'] / 10
    
    df.to_csv(preprocessed_dataset.path, index=False)
    
    
@dsl.pipeline(name="sample-pipeline")
def sample_pipeline():
    
    create_dataset_op = create_dataset()
    operations_op = operations(dataset=create_dataset_op.outputs['dataset'])
    
    
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