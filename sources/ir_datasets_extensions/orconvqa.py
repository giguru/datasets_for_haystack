import io
import os
import platform
import pickle as pkl
import json
import gzip
import joblib
import shutil
import logging
import ir_datasets

from gzip import GzipFile
from ir_datasets.formats.base import BaseQrels, BaseQueries, BaseDocs
from ir_datasets.util import GzipExtract, Download

from haystack.schema import Document, Label
from .util import ConverseDownloadConfig
from .schema import Qrel
from ir_datasets.datasets.base import Dataset, YamlDocumentation

NAME = 'orconvqa'

logger = logging.getLogger(__name__)

# TODO Find data use agreemnt
DATA_USE_AGREEMENT = ""

__all__ = ['collection', 'subsets']


def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)

def load_pickle(source_file: GzipFile):
    if not os.path.isfile(source_file.name):
        raise FileNotFoundError(f'Zipped file not found: {source_file.name}')
    fname = os.path.splitext(source_file.name)[0]
    if not os.path.isfile(fname):
        logger.info(f"Unzipping file: {fname}")
        gunzip_shutil(source_file.name, fname)

    # Fix reading large pickle files on MAC systems
    if platform.system() == "Darwin":
        bytes_in = bytearray(0)
        max_bytes = 2 ** 31 - 1
        input_size = os.path.getsize(fname)
        with open(fname, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        return pkl.loads(bytes_in)

    with open(fname, 'rb') as handle:
        return joblib.load(handle)

def create_inv_passage_id_index(passage_ids):
    # TODO this seems like a slow way to do this
    passage_id_to_idx = {}
    for i, pid in enumerate(passage_ids):
        passage_id_to_idx[pid] = i
    return passage_id_to_idx


class ORConvQAQrels(BaseQrels):
    def __init__(self, dlc):
        self._dlc = dlc

    def qrels_iter(self):
        with self._dlc.stream() as f:
            dict = json.load(f)
            for query_id in dict:
                document_id = next(iter(dict[query_id].keys()))
                score = dict[query_id][document_id]
                yield Qrel(query_id=query_id,
                           doc_id=document_id,
                           relevance=int(score),
                           iteration=0)

    def get_json_file(self) -> dict:
        with self._dlc.stream() as f:
            return json.load(f)

    def qrels_defs(self):
        raise NotImplementedError()

    def qrels_path(self):
        raise NotImplementedError()


class ORConvQAQueries(BaseQueries):
    def __init__(self, dlc, qrels: ORConvQAQrels):
        self._dlc = dlc
        self._qrels = qrels

    def queries_iter(self) -> Label:
        qrels = self._qrels.get_json_file()

        with self._dlc.stream() as f:
            f = io.TextIOWrapper(f)
            for line in f:
                if line == '\n':
                    continue  # ignore blanks
                json_object = json.loads(line)
                """
                Each json object in ORConvQA has the following structure:
                qid: str
                question: str
                rewrite: str
                followup: str 'n' or 'y'
                yesno: str 'y' or 'n'
                evidences: list[str]
                answer: dict
                    text: str
                    answer_start: 0
                history: list[str]
                retriever_labels: list[int] Either zero or one.
                """
                try:
                    q_doc_rel = qrels[json_object['qid']]
                    if len(q_doc_rel.keys()) > 1:
                        logger.warning(f'Found qrel with multiple docs, golden passage is unknown, assuming first. QID={json_object["qid"]}. Qrels={str(q_doc_rel.keys())}')
                except:
                    logger.warning(f'QID {json_object["qid"]} not found in qrels, skipping question')
                    continue

                document_id = next(iter(q_doc_rel.keys()))

                yield Label(question=json_object['rewrite'],
                            answer=json_object['answer']['text'],
                            document_id=document_id,
                            no_answer=(json_object["answer"]['text'] == 'CANNOTANSWER'),
                            offset_start_in_doc=json_object["answer"]['answer_start'],
                            is_correct_answer=True,
                            is_correct_document=True,
                            origin=f.name,
                            # original_question=json_object['question'],
                            # history=[previous_question['question'] for previous_question in json_object['history']]
                            )


class OrConvQADocs(BaseDocs):
    def __init__(self, docs_dlc, passage_id_to_idx: dict, passage_reps):
        self._dlc = docs_dlc
        self._passage_id_to_idx = passage_id_to_idx
        self._passage_reps = passage_reps

    def docs_iter(self) -> Document:
        with self._dlc.stream() as f:
            f = io.BufferedReader(f)
            for line in f:
                if line == '\n':
                    continue # ignore blanks
                json_object = json.loads(line)
                doc_id = json_object['id']
                passage_idx = self._passage_id_to_idx[doc_id]
                yield Document(id=doc_id,
                               text=json_object['text'],
                               meta={k: v for k, v in json_object.items() if k not in ['text', 'id']},
                               embedding=self._passage_reps[passage_idx])


def _init():
    documentation = YamlDocumentation('docs/orconvqa.yaml')
    base_path = ir_datasets.util.home_path() / NAME
    dlc = ConverseDownloadConfig.context(NAME, base_path, dua=DATA_USE_AGREEMENT)

    # ORConvQA have precomputed passage representations. These are provided, because the full corpus contains 11M+
    # passages and embeddings them using one GPU (GeForce 1080Ti) takes 100 days...
    with GzipExtract(dlc['passage_ids']).stream() as f:
        passage_ids = load_pickle(f)
    with GzipExtract(dlc['representations']).stream() as f:
        passage_reps = load_pickle(f)
    # To time efficiently find the passage representation given a passage id, create an inverse dictionary
    passage_id_to_idx = create_inv_passage_id_index(passage_ids=passage_ids)

    # Documents sets
    full_collection = OrConvQADocs(GzipExtract(dlc['docs']), passage_id_to_idx=passage_id_to_idx, passage_reps=passage_reps)
    dev_collection = OrConvQADocs(GzipExtract(dlc['dev/docs']), passage_id_to_idx=passage_id_to_idx, passage_reps=passage_reps)

    # Qrels
    qrels = ORConvQAQrels(GzipExtract(dlc['qrels']))

    subsets = {
        'dev': Dataset(full_collection, ORConvQAQueries(dlc['dev/queries'], qrels), qrels),
        'train': Dataset(full_collection, ORConvQAQueries(dlc['train/queries'], qrels), qrels),
        'test': Dataset(full_collection, ORConvQAQueries(dlc['test/queries'], qrels), qrels),
    }

    # Register data sets
    ir_datasets.registry.register(NAME, Dataset(full_collection, documentation('_')))
    for s in sorted(subsets):
        ir_datasets.registry.register(f'{NAME}/{s}', Dataset(subsets[s], documentation(s)))

    return full_collection, subsets

collection, subsets = _init()
