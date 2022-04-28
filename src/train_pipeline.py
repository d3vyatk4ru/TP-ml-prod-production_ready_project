
import json
import logging
import click

import pandas as pd

from src.entity.train_pipeline_params import TrainingPipelineParams
from src.data.make_dataset import (
    read_dataset,
    split_train_val_data,
)

from src.features.make_features import (
    extract_target,
    drop_target,
    build_transformer,
    make_features,
)

from src.models.model_fit_predict import (
    train_model,
    create_inference_pipeline,
    predict_model,
    evaluate_model,
)

from src.entity.train_pipeline_params import (
    read_training_pipelineparams, 
) 


logger = logging.getLogger(__name__)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipelineparams(config_path)

    return run_train_pipeline(training_pipeline_params)

def run_train_pipeline(training_pipeline_params: TrainingPipelineParams):
    ''' Launching pipeline for training ml model '''
    
    logger.info(f'Start train pipeline with {training_pipeline_params}...')

    data: pd.DataFrame = read_dataset(training_pipeline_params.input_data_path)

    logger.info(f'data.shape is equal {data.shape}...')

    train_df, valid_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    logger.info('data splitted to train and valid...')

    train_target = extract_target(
        train_df, training_pipeline_params.feature_params
    )
    train_df = drop_target(
        train_df, training_pipeline_params.feature_params
    )

    valid_target = extract_target(
        valid_df, training_pipeline_params.feature_params
    )
    valid_df = drop_target(
        valid_df, training_pipeline_params.feature_params
    )

    logger.info('The target column was write to other list')
    logger.info(f"train_df.shape is equal {train_df.shape}")
    logger.info(f"val_df.shape is equal {valid_df.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)

    transformer.fit(train_df)

    train_features = make_features(transformer, train_df)

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    inference_pipeline = create_inference_pipeline(model, transformer)

    pred = predict_model(
        inference_pipeline,
        valid_df,
    )

    metrics = evaluate_model(
        pred,
        valid_target,
    )

    with open(training_pipeline_params.metric_path, 'w') as file_metrics:
        json.dump(metrics, file_metrics)

    return metrics

@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)

if __name__ == "__main__":
    train_pipeline_command()