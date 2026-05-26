import jax
import optax

from jax_prism._typing import PyTree


def _key_to_str(key) -> str:
    """Convert a JAX path key to a string.

    Handles different key types:
    - DictKey: has .key attribute
    - SequenceKey: has .idx attribute
    - GetAttrKey: has .name attribute
    - FlattenedIndexKey: has .key attribute
    """
    if hasattr(key, "key"):
        return str(key.key)
    elif hasattr(key, "idx"):
        return str(key.idx)
    elif hasattr(key, "name"):
        return str(key.name)
    else:
        return str(key)


def freeze_params_by_pattern(
    optimizer: optax.GradientTransformation,
    params: PyTree,
    freeze_patterns: list[str]
) -> optax.GradientTransformation:
    """Wrap optimizer to freeze params matching any pattern.

    Args:
        optimizer: Base optimizer to use for unfrozen params.
        params: Parameter tree (needed for structure).
        freeze_patterns: List of substrings. Params with paths
            containing any pattern are frozen.

    Returns:
        New optimizer that zeros gradients for frozen params.
    """
    def label_fn(path, _):
        path_str = "/".join(_key_to_str(p) for p in path)
        for pattern in freeze_patterns:
            if pattern in path_str:
                return "frozen"
        return "trainable"
    
    param_labels = jax.tree_util.tree_map_with_path(label_fn, params)
    
    return optax.multi_transform(
        transforms={
            "frozen": optax.set_to_zero(),
            "trainable": optimizer,
        },
        param_labels=param_labels,
    )