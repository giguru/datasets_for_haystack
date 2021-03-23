import io
import os
import json
import ir_datasets
from ir_datasets.formats.base import BaseQrels, BaseQueries, BaseDocs
from ir_datasets.util import ZipExtract, Download

from .util import ConverseDownloadConfig
from .schema import Qrel
from haystack.schema import Document, Label
from ir_datasets.datasets.base import Dataset, YamlDocumentation

NAME = 'canard'
__all__ = ['collection', 'subsets']

DATA_USE_AGREEMENT = ""

class CanardQueries(BaseQueries):
    def __init__(self, dlc):
        """
        The CANARD comes with the following JSON files: dev.json, train.json and test.json

        :param dlc:
        """
        self._dlc = dlc

    def queries_iter(self) -> Label:
        with self._dlc.stream() as f:
            # CANARD is a small files, so no need to buffer by reading entry by entry
            contents = json.loads(f.read())
            for json_object in contents:
                """
                The JSON object in CANARD has the following keys:
                History: str[] (Both questions and answers.)
                QuAC_dialog_id: str From the dataset Question Answering in Context (QuAC)
                Question: str
                Question_no: int
                Rewrite: str
                """
                yield Label(question=json_object['Question'],
                            answer=json_object['Rewrite'],
                            id=json_object['Question_no'],
                            is_correct_answer=True,
                            is_correct_document=True,
                            no_answer=False,
                            origin=f.name
                            )

def _init():
    documentation = YamlDocumentation('docs/canard.yaml')
    base_path = ir_datasets.util.home_path() / NAME
    dlc = ConverseDownloadConfig.context(NAME, base_path, dua=DATA_USE_AGREEMENT)
    placeholder_docs = BaseDocs()
    qrels = BaseQrels()

    subsets = {
        'dev': Dataset(placeholder_docs, CanardQueries(ZipExtract(dlc['queries'], 'CANARD_Release/dev.json')), qrels),
        'test': Dataset(placeholder_docs, CanardQueries(ZipExtract(dlc['queries'], 'CANARD_Release/test.json')), qrels),
        'train': Dataset(placeholder_docs, CanardQueries(ZipExtract(dlc['queries'], 'CANARD_Release/train.json')), qrels),
    }

    # Register data sets
    for key in sorted(subsets):
        ir_datasets.registry.register(f'{NAME}/{key}', Dataset(subsets[key], documentation('_')))

    return placeholder_docs, subsets

collection, subsets = _init()