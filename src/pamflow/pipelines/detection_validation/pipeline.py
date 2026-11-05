from kedro.pipeline import Pipeline, node

from .nodes import (
    find_model_threshold,
    get_validation_status,
    update_observations,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_validation_status,
                inputs="validation_results@PartitionedDataset",
                outputs="segment_validation_results@pandas",
                name="get_validation_status_node",
            ),
            node(
                func=find_model_threshold,
                inputs=[
                    "segment_validation_results@pandas",
                    "params:detection_validation_parameters",
                ],
                outputs=[
                    "species_confidence_thresholds@pandas",
                    "species_confusions@pandas",
                    "threshold_plots@PartitionedDataset",
                ],
                name="find_model_threshold_node",
            ),
            node(
                func=update_observations,
                inputs=[
                    "observations@pamDP",
                    "species_confidence_thresholds@pandas",
                    "species_confusions@pandas",
                    "segment_validation_results@pandas",
                    "params:detection_validation_parameters",
                ],
                outputs="observations_thresholded@pandas",
                name="update_observations_node",
            ),
        ]
    )
