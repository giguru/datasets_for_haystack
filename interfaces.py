import logging
from typing import Optional, Union
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.base import BaseDocumentStore
import ir_datasets
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['load_dataset_into_documentstore']


def load_dataset_into_documentstore(dataset_name: str,
                                    document_store: Union[BaseDocumentStore, FAISSDocumentStore],
                                    verbose: bool = True,
                                    index: Optional[str] = None,
                                    batch_size: int = 1024):
    dataset = ir_datasets.load(dataset_name)

    has_faiss_index = document_store.faiss_index is not None
    do_in_batches = batch_size > 0  # If the batch size is zero, write all documents in one go
    vector_id = document_store.faiss_index.ntotal if has_faiss_index else None

    if verbose:
        logger.info(f'Started loading documents into document store...')
    docs_to_write_in_sql = []
    batch_iter = 0

    for doc in dataset.docs_iter():
        if has_faiss_index and doc.embedding is not None:
            document_store.faiss_index.add(np.array([doc.embedding], dtype="float32"))
            doc.meta["vector_id"] = vector_id
            vector_id += 1
        docs_to_write_in_sql.append(doc)

        # calling write_documents with batches is faster than calling write_documents for each documents.
        # Probably, since FAISSDocumentStore uses SQL, which comes with connection times
        if len(docs_to_write_in_sql) == batch_size and do_in_batches:
            document_store.write_documents(docs_to_write_in_sql, index=index)
            docs_to_write_in_sql = []  # clear array
            batch_iter += 1

    # add leftover docs to document store
    if len(docs_to_write_in_sql) > 0:
        document_store.write_documents(docs_to_write_in_sql, index=index)

    if verbose:
        logger.info(f'Indexed {batch_size * batch_iter + len(docs_to_write_in_sql)} docs')
