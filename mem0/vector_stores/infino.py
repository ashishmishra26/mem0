import json
import logging
import os
from typing import Dict, List, Optional

from pydantic import BaseModel

try:
    import infino
    import pyarrow as pa
except ImportError:
    raise ImportError(
        "The 'infino' library is required. Install it with: pip install infino"
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

# infino requires vector dimensions in [16, 4096].
_MIN_DIM = 16

# mem0 distance strategy -> infino vector index metric.
_METRIC = {"cosine": "cosine", "euclidean": "l2", "inner_product": "dot"}


class OutputData(BaseModel):
    id: Optional[str]  # memory id
    score: Optional[float]  # similarity (higher = better)
    payload: Optional[Dict]  # metadata


def _sql_str(value: str) -> str:
    """Escape a single-quoted SQL string literal."""
    return value.replace("'", "''")


def _rows(result) -> List[dict]:
    """infino returns an empty result with no schema, on which pyarrow's
    to_pylist() raises; treat that as no rows."""
    try:
        return result.to_pylist()
    except ValueError:
        return []


class InfinoVectorStore(VectorStoreBase):
    """
    Infino vector store for mem0.

    Infino is an embedded, object-storage-native retrieval engine (SQL + BM25 +
    vector + hybrid over Apache Parquet). Because mem0's ``search`` hands the
    store BOTH the query text and its embedding, this adapter runs infino's
    native ``hybrid_search`` (BM25 + vector, reciprocal-rank fused) as the
    primary search path -- so mem0 gets hybrid recall in one call, with the
    index living on a local path or object storage (s3://, az://, gs://) and no
    separate cluster to operate. ``keyword_search`` exposes pure BM25.
    """

    def __init__(
        self,
        collection_name: str = "mem0",
        path: Optional[str] = None,
        embedding_model_dims: int = 1536,
        n_cent: int = 1,
        distance_strategy: str = "cosine",
    ):
        if embedding_model_dims < _MIN_DIM:
            raise ValueError(f"infino requires embedding_model_dims >= {_MIN_DIM}, got {embedding_model_dims}")

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.n_cent = n_cent
        self.metric = _METRIC.get(distance_strategy, "cosine")
        # Durable path (needed for update/delete). Also accepts s3://|az://|gs:// URIs.
        self.uri = path or os.path.join(os.path.expanduser("~"), ".mem0", "infino")
        if "://" not in self.uri:
            os.makedirs(self.uri, exist_ok=True)

        self.db = infino.connect(self.uri)
        self.table = None
        self.create_col(collection_name, embedding_model_dims, distance_strategy)

    # ---------- collection lifecycle ----------

    def _schema(self) -> "pa.Schema":
        return pa.schema(
            [
                pa.field("id", pa.large_utf8(), nullable=False),
                pa.field("text", pa.large_utf8(), nullable=False),
                pa.field("payload", pa.large_utf8(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_model_dims), nullable=False),
            ]
        )

    def create_col(self, name, vector_size, distance="cosine"):
        """Create the collection, or open it if it already exists."""
        self.collection_name = name
        if name in self.db.list_tables():
            self.table = self.db.open_table(name)
            return
        metric = _METRIC.get(distance, self.metric)
        indexes = infino.IndexSpec().fts("text").vector("vector", vector_size, self.n_cent, metric)
        self.table = self.db.create_table(name, self._schema(), indexes)
        logger.info(f"Created infino collection '{name}' (dim={vector_size}, metric={metric})")

    def list_cols(self):
        return self.db.list_tables()

    def delete_col(self):
        self.db.drop_table(self.collection_name, purge=True)
        self.table = None

    def col_info(self):
        rows = self._safe_rows(lambda: self.db.query_sql(f"SELECT COUNT(*) AS n FROM {self.collection_name}"))
        return {"name": self.collection_name, "count": rows[0]["n"] if rows else 0}

    def reset(self):
        logger.warning(f"Resetting infino collection '{self.collection_name}'")
        try:
            self.delete_col()
        except Exception:
            pass
        self.create_col(self.collection_name, self.embedding_model_dims, self.metric)

    # ---------- write ----------

    def insert(self, vectors, payloads=None, ids=None):
        payloads = payloads or [{} for _ in vectors]
        ids = ids or [str(i) for i in range(len(vectors))]
        rows = []
        for vector, payload, _id in zip(vectors, payloads, ids):
            rows.append(
                {
                    "id": str(_id),
                    "text": str(payload.get("data", "")),  # mem0 stores the memory text under "data"
                    "payload": json.dumps(payload),
                    "vector": list(vector),
                }
            )
        if rows:
            self.table.append(rows)
        logger.debug(f"Inserted {len(rows)} vectors into '{self.collection_name}'")

    def update(self, vector_id, vector=None, payload=None):
        # mem0 re-embeds on update and passes the fresh vector; the stored vector
        # column isn't read back via SQL, so update is an upsert: delete + insert.
        if vector is None:
            logger.warning(f"infino update for '{vector_id}' without a vector; skipping (mem0 normally re-embeds)")
            return
        existing = self.get(vector_id)
        new_payload = payload if payload is not None else (existing.payload if existing else {})
        self.delete(vector_id)
        self.insert([list(vector)], [new_payload or {}], [str(vector_id)])

    def delete(self, vector_id):
        self.table.delete(f"id = '{_sql_str(str(vector_id))}'")

    # ---------- read / search ----------

    def search(self, query, vectors, top_k=5, filters=None):
        """Semantic (vector) search returning cosine SIMILARITY (higher = better).

        mem0 runs its own hybrid fusion: it calls this (semantic) plus
        keyword_search (BM25) and combines them. Infino supplies both halves
        natively, so mem0 gets hybrid recall from a single embedded engine.
        infino's vector_search returns a distance, so convert to similarity per
        mem0's contract (cosine: 1 - distance)."""
        fetch_k = top_k if not filters else max(top_k * 10, 100)
        rows = self._safe_rows(
            lambda: self.table.vector_search("vector", list(vectors), fetch_k, projection=["id", "payload", "score"])
        )
        out = self._outputs(rows, filters, top_k)
        for o in out:
            if o.score is not None:
                o.score = self._to_similarity(o.score)
        return out

    def keyword_search(self, query, top_k=5, filters=None):
        """Native BM25 full-text search (mem0's optional keyword path)."""
        fetch_k = top_k if not filters else max(top_k * 10, 100)
        rows = self._safe_rows(
            lambda: self.table.bm25_search("text", (query or "").strip(), fetch_k, projection=["id", "payload", "score"])
        )
        return self._outputs(rows, filters, top_k)

    def get(self, vector_id):
        rows = self._safe_rows(
            lambda: self.db.query_sql(
                f"SELECT id, payload FROM {self.collection_name} WHERE id = '{_sql_str(str(vector_id))}' LIMIT 1"
            )
        )
        return self._row_to_output(rows[0]) if rows else None

    def list(self, filters=None, top_k=None):
        cap = (top_k or 100) if not filters else max((top_k or 100) * 10, 1000)
        rows = self._safe_rows(
            lambda: self.db.query_sql(f"SELECT id, payload FROM {self.collection_name} LIMIT {cap}")
        )
        out = [self._row_to_output(r) for r in rows]
        if filters:
            out = [o for o in out if self._match(o.payload, filters)]
        if top_k:
            out = out[:top_k]
        # mem0 expects list() to return a list of result batches.
        return [out]

    # ---------- helpers ----------

    @staticmethod
    def _safe_rows(thunk) -> List[dict]:
        """Run a query/search that may match nothing. infino raises on an empty
        result (no schema) from the call itself, so guard the call and treat it
        as no rows."""
        try:
            result = thunk()
        except ValueError:
            return []
        return _rows(result)

    def _row_to_output(self, row: dict) -> OutputData:
        payload = row.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        score = row.get("score")
        return OutputData(id=str(row.get("id")), score=float(score) if score is not None else None, payload=payload or {})

    def _outputs(self, rows, filters, top_k) -> List[OutputData]:
        out = [self._row_to_output(r) for r in rows]
        if filters:
            out = [o for o in out if self._match(o.payload, filters)]
        return out[:top_k]

    @staticmethod
    def _clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    def _to_similarity(self, score: float) -> float:
        """Convert infino's vector-search score to a [0,1] similarity
        (higher = better), matching mem0's scoring contract. Clamped, since
        infino's cosine score can fall outside a plain [0,1] distance range."""
        if self.metric == "l2":
            return 1.0 / (1.0 + abs(score))
        if self.metric == "dot":
            return self._clamp(score)
        return self._clamp(1.0 - score)  # cosine

    @staticmethod
    def _match(payload: Optional[Dict], filters: Dict) -> bool:
        if not filters:
            return True
        payload = payload or {}
        for key, value in filters.items():
            if payload.get(key) != value:
                return False
        return True
