import os
import sagemaker
import boto3
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.estimator import SKLearn

def create_sagemaker_pipeline():
    pipeline_session = PipelineSession()
    try:
        role = sagemaker.get_execution_role()
    except Exception:
        iam = boto3.client("iam")
        role = iam.get_role(RoleName="LabRole")["Role"]["Arn"]

    default_bucket = pipeline_session.default_bucket()
    base_job_prefix = 'credit-score-pipeline'
    s3_raw_data = f"s3://{default_bucket}/{base_job_prefix}/raw/data_C.csv"
    instance_type = 'ml.m5.medium'

    sklearn_processor = SKLearnProcessor(
        framework_version="1.4-2",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
        base_job_name=f"{base_job_prefix}-ingest"
    )

    step_process = ProcessingStep(
        name="CreditDataIngestion",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=s3_raw_data, destination="/opt/ml/processing/input")
        ],
        outputs=[
            ProcessingOutput(output_name='train_data', source='/opt/ml/processing/output/train')
        ],
        code='src/data_ingestion.py'
    )

    sklearn_estimator = SKLearn(
        entry_point="train.py",
        source_dir='src',
        framework_version="1.4-2",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
        base_job_name=f'{base_job_prefix}-train'
    )

    step_train = TrainingStep(
        name="CreditModelTraining",
        estimator=sklearn_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    step_eval = ProcessingStep(
        name="CreditModelEvaluation",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs['train_data'].S3Output.S3Uri,
                destination='/opt/ml/processing/test'
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source='/opt/ml/processing/evaluation')
        ],
        code='src/evaluation.py',
        property_files=[evaluation_report]
    )

    cond_GTE = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classifcation_metrics.f1_macro.value"
        ),
        right=0.65
    )

    step_cond = ConditionStep(
        name="CheckCreditModelF1Macro",
        conditions=[cond_GTE],
        if_steps=[],
        else_steps=[]
    )

    pipeline = Pipeline(
        name="CreditScoreProductionPipeline",
        steps=[step_process, step_train, step_eval, step_cond]
    )

    return pipeline

if __name__ == "__main__":
    pipeline = create_sagemaker_pipeline()
    try:
        role_arn = sagemaker.get_execution_role()
    except Exception:
        iam = boto3.client("iam")
        role_arn = iam.get_role(RoleName="LabRole")["Role"]["Arn"]
    pipeline.upsert(role_arn=role_arn)
    execution = pipeline.start()