from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InfinoConfig(BaseModel):
    collection_name: str = Field("mem0", description="Name of the collection (infino table)")
    path: Optional[str] = Field(
        None,
        description="Local path or object-storage URI (s3://, az://, gs://) for the infino catalog",
    )
    embedding_model_dims: int = Field(1536, description="Dimension of the embedding vector (infino requires >= 16)")
    n_cent: int = Field(1, description="IVF centroids for the vector index; 1 = exact (right for memory-scale stores)")
    distance_strategy: str = Field(
        "cosine", description="Distance metric. Options: 'cosine', 'euclidean', 'inner_product'"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_distance_strategy(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        strategy = values.get("distance_strategy")
        if strategy and strategy not in ["cosine", "euclidean", "inner_product"]:
            raise ValueError(
                "Invalid distance_strategy. Must be one of: 'cosine', 'euclidean', 'inner_product'"
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = ConfigDict(arbitrary_types_allowed=True)
