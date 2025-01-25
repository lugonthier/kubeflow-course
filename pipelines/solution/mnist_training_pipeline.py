import kfp.dsl as dsl_v1
from kfp.v2 import dsl
from kfp import compiler
from pipelines.solution.components import load_and_split_data, preprocess_data, train_model, evaluate_model


@dsl.pipeline(name="mnist-training-pipeline")
def training_pipeline():
    load_and_split_data_op = load_and_split_data()
    

    preprocess_train_data_op = preprocess_data(
        dataset=load_and_split_data_op.outputs['train_dataset'],
        size=load_and_split_data_op.outputs['train_size']
    )
    
    preprocess_test_data_op = preprocess_data(
        dataset=load_and_split_data_op.outputs['test_dataset'],
        size=load_and_split_data_op.outputs['test_size']
    )
    
    
    train_model_op = train_model(preprocess_train_data_op.outputs['preprocessed_dataset'])
    
    evaluate_model(
        test_dataset=preprocess_test_data_op.outputs['preprocessed_dataset'],
        model_artifact=train_model_op.outputs['model_artifact']
    )
    
    
    
    

if __name__=="__main__":    
    from pipelines.kfp_client import KFPClientManager

    kfp_client_manager = KFPClientManager(
        api_url="https://deploykf.example.com:8443/pipeline",
        skip_tls_verify=True,

        dex_username="admin@example.com",
        dex_password="admin",

        dex_auth_type="local",
    )

    kfp_client = kfp_client_manager.create_kfp_client()

    compiler.Compiler(mode=dsl_v1.PipelineExecutionMode.V2_COMPATIBLE).compile(
        pipeline_func=training_pipeline,
        package_path='classif_pipeline.yaml'
    )

    kfp_client.create_run_from_pipeline_package(
        pipeline_file='classif_pipeline.yaml',
        arguments={},
        namespace='team-1'
    )