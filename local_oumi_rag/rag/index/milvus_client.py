from __future__ import annotations

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


class MilvusIndex:
    def __init__(self, uri: str, collection_name: str, dim: int):
        connections.connect(alias="default", uri=uri)
        self.collection_name = collection_name
        self.dim = dim
        self.collection = self._ensure_collection()

    def _ensure_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="rel_path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="start_line", dtype=DataType.INT64),
                FieldSchema(name="end_line", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
        )
        collection = Collection(self.collection_name, schema=schema)
        collection.create_index(field_name="dense", index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}})
        return collection

    def reset(self) -> None:
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        self.collection = self._ensure_collection()

    def insert(self, *, ids, rel_paths, doc_types, start_lines, end_lines, texts, vectors) -> None:
        safe_texts = [t[:65000] for t in texts]
        self.collection.insert([ids, rel_paths, doc_types, start_lines, end_lines, safe_texts, vectors])
        self.collection.flush()

    def search(self, query_vector: list[float], top_k: int = 20):
        self.collection.load()
        res = self.collection.search(
            data=[query_vector],
            anns_field="dense",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["rel_path", "doc_type", "start_line", "end_line", "text"],
        )
        return res[0]
