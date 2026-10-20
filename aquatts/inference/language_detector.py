"""Bind the selected upstream splitter to a private FastText detector.

The upstream APIs have no detector argument. Rebind their Python functions to
private globals instead of replacing fast_langdetect's process-wide default.
No upstream source file or existing function globals are modified.
"""
from types import FunctionType, SimpleNamespace
import threading


def _bind(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def bind_segmenter(get_texts, detector):
    from split_lang.detect_lang import detector as detection
    from split_lang.split import splitter
    from aquatts.upstream import GPTSoVITSConfigurationError

    required = ("fast_lang_detect", "detect_lang_combined", "possible_detection_list")
    if (not isinstance(get_texts, FunctionType)
            or "LangSplitter" not in get_texts.__code__.co_names
            or any(not isinstance(getattr(detection, name, None), FunctionType) for name in required)
            or getattr(splitter, "detect_lang_combined", None) is not detection.detect_lang_combined
            or getattr(splitter, "possible_detection_list", None) is not detection.possible_detection_list):
        raise GPTSoVITSConfigurationError("Unsupported LangSegmenter/split-lang detector interface")
    lock = threading.RLock()
    def detect(*args, **kwargs):
        with lock:
            return detector.detect(*args, **kwargs)
    private_detection = dict(vars(detection), fast_langdetect=SimpleNamespace(detect=detect))
    for name in required:
        private_detection[name] = _bind(getattr(detection, name), private_detection)
    private_splitter = dict(vars(splitter))
    for name in required[1:]:
        private_splitter[name] = private_detection[name]
    methods = {}
    for name, method in vars(splitter.LangSplitter).items():
        if isinstance(method, FunctionType):
            if method.__closure__:
                raise GPTSoVITSConfigurationError("Unsupported LangSplitter method closure")
            methods[name] = _bind(method, private_splitter)
    local_splitter = type("InstanceLangSplitter", (splitter.LangSplitter,), methods)
    private_splitter["LangSplitter"] = local_splitter
    namespace = dict(get_texts.__globals__, LangSplitter=local_splitter)
    return _bind(get_texts, namespace)
