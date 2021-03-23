
class Qrel:
    def __init__(self, query_id: str, doc_id: str, relevance: int, iteration):
        self.query_id = query_id
        self.doc_id = doc_id
        self.iteration = iteration
        self.relevance = relevance