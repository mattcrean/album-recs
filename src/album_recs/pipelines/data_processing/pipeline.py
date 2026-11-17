"""
Pipeline 'data_processing' definition.
Generated using Kedro 1.8.0
"""

from kedro.pipeline import Node, Pipeline

from .nodes import preprocess_playlists_dump, preprocess_tracks


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=preprocess_tracks,
                inputs="tracks_features",
                outputs="preprocessed_tracks",
                name="preprocess_tracks_node",
            ),
            Node(
                func=preprocess_playlists_dump,
                inputs=["playlists_dump", "params:intermediate_dir"],
                outputs=[
                    "preprocessed_playlists",
                    "preprocessed_sql_tracks",
                    "preprocessed_sql_albums",
                    "preprocessed_playlist_tracks",
                ],
                name="preprocess_playlists_dump_node",
            ),
        ]
    )
