from dataclasses import dataclass

from src.features.custom_transformer import CustomTransformes

from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams

from marshmallow_dataclass import class_schema
import yaml


@dataclass
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    custom_transformer_params: CustomTransformes


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipelineparams(path: str):
    with open(path, 'r') as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))