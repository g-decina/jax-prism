"""Serialization exceptions."""

class CheckpointError(Exception):
    """Base exception for checkpoint operations."""
    pass

class CheckpointVersionError(CheckpointError):
    """Checkpoint format version is newer than this library supports.
    
    This typically means the checkpoint was saved with a newer version 
    of jax-prism. Upgrade the library or use a compatible checkpoint.
    """
    
    def __init__(self, found_version: int, max_supported: int) -> None:
        self.found_version = found_version
        self.max_supported = max_supported
        
        super().__init__(
            f"Checkpoint format version {found_version} is newer than "
            f"supported version {max_supported}. Upgrade jax-prism."
        )
        
class CheckpointCorruptedError(CheckpointError):
    """Checkpoint is missing required files or has invalid structure."""
    pass

class UnknownModelTypeError(CheckpointError):
    """Model type string not found in registry.
    
    This means the checkpoint references a model type that isn't
    registered. Either the model class wasn't imported, or the
    checkpoint is from an incompatible library version.
    """
    
    def __init__(self, model_type: str, available: list[str]) -> None:
        self.model_type = model_type
        self.available = available
        super().__init__(
            f"Unknown model type '{model_type}'. "
            f"Available types: {available}"
        )