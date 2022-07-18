from pathlib import Path
from typing import Dict, List, Any

import yaml
from pyannote.audio.tasks.segmentation.multilabel import MultiLabelSegmentation
from pyannote.audio.utils.preprocessors import DeriveMetaLabels
from pyannote.database import FileFinder, get_protocol, ProtocolFile
from pyannote.database.protocol.protocol import Preprocessor
from pyannote.database.util import LabelMapper


class ProcessorChain:

    def __init__(self, preprocessors: List[Preprocessor], key: str):
        self.procs = preprocessors
        self.key = key

    def __call__(self, file: ProtocolFile):
        file_cp: Dict[str, Any] = abs(file)
        for proc in self.procs:
            out = proc(file_cp)
            file_cp[self.key] = out

        return out


CLASSES = {"vtcdebug": {'classes': ["READER", "AGREER", "DISAGREER"],
                        'unions': {"COMMENTERS": ["AGREER", "DISAGREER"]},
                        'intersections': {}},
           "basal_voice": {'classes': ["P", "NP"],
                           'unions': {},
                           'intersections': {}},
           "babytrain": {'classes': ["MAL", "FEM", "CHI", "KCHI"],
                         'unions': {"SPEECH": ["MAL", "FEM", "CHI", "KCHI"]},
                         'intersections': {}}
           }


def build_protocol(protocol, classes):
    classes_kwargs = CLASSES[classes]
    vtc_preprocessor = DeriveMetaLabels(**classes_kwargs)
    preprocessors = {
        "audio": FileFinder(),
        "annotation": vtc_preprocessor
    }
    if classes == "babytrain":
        with open(Path(__file__).parent / "data/babytrain_mapping.yml") as mapping_file:
            mapping_dict = yaml.safe_load(mapping_file)["mapping"]
        preprocessors["annotation"] = ProcessorChain([
            LabelMapper(mapping_dict, keep_missing=True),
            vtc_preprocessor
        ], key="annotation")
    return get_protocol(protocol, preprocessors=preprocessors)


protocol = build_protocol("BABYTRAIN.SpeakerDiarization.babytrain", "babytrain")
task = MultiLabelSegmentation(protocol, duration=2.00)
task.setup()
