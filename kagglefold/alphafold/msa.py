from typing import Mapping, Any

import numpy as np
import tensorflow as tf

from alphafold.model.features import FeatureDict
from alphafold.model.tf import shape_placeholders

NUM_RES = shape_placeholders.NUM_RES
NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES


def make_fixed_size(
    protein: Mapping[str, Any],
    shape_schema,
    msa_cluster_size: int,
    extra_msa_size: int,
    num_res: int,
    num_templates: int = 0,
) -> FeatureDict:
    """Guess at the MSA and sequence dimensions to make fixed size."""

    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)

        schema = shape_schema[k]

        assert len(shape) == len(schema), (
            f"Rank mismatch between shape and shape schema for {k}: "
            f"{shape} vs {schema}"
        )
        pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
        padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]

        if padding:
            # TODO: alphafold's typing is wrong
            protein[k] = tf.pad(v, padding, name=f"pad_to_fixed_{k}")
            protein[k].set_shape(pad_size)
    return {k: np.asarray(v) for k, v in protein.items()}
