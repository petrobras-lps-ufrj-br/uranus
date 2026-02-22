__all__ = [
    "CogniteConnectionError",
]



class CogniteConnectionError(Exception):
    """Raised when a specified dataset is not found."""
    def __init__(self, name):
        """Set the error message with the dataset name."""
        message = f"Could not connect to Cognite project: {name}"
        super().__init__(message)
        
