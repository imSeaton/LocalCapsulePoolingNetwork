"""
Constants and enums for dataset configuration.
"""
from enum import Enum
from typing import List


class SquashFuncType(Enum):
    """Squashing function types for different dataset categories."""
    BIOLOGICAL = ['DD', 'NCI109', 'NCI1', 'MUTAG', 'ENZYMES', 'FRANKENSTEIN', 'REDDIT-BINARY']
    SOCIAL = ['PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-MULTI', 'COLLAB']

    @classmethod
    def get_squash_func_name(cls, dataset_name: str) -> str:
        """Return the squash function name for a given dataset."""
        if dataset_name in cls.BIOLOGICAL.value:
            return 'squash_1'
        elif dataset_name in cls.SOCIAL.value:
            return 'squash_2'
        else:
            raise ValueError(f'Unsupported dataset: {dataset_name}')

    @classmethod
    def is_biological(cls, dataset_name: str) -> bool:
        return dataset_name in cls.BIOLOGICAL.value

    @classmethod
    def is_social(cls, dataset_name: str) -> bool:
        return dataset_name in cls.SOCIAL.value


# Alias for backward compatibility
DATASET_BIOLOGICAL = SquashFuncType.BIOLOGICAL.value
DATASET_SOCIAL = SquashFuncType.SOCIAL.value
