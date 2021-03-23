import ir_datasets
import os
from ir_datasets.util.download import _DownloadConfig

__all__ = ['ConverseDownloadConfig']

_dir_path = os.path.dirname(os.path.realpath(__file__))
_file = _dir_path + '/added_datasets.yaml'
_yaml = ir_datasets.lazy_libs.yaml()
_data = open(_file, 'rb').read()
_contents = _yaml.load(_data, Loader=_yaml.BaseLoader)

ConverseDownloadConfig = _DownloadConfig(file=_file, base_path=None, contents=_contents)