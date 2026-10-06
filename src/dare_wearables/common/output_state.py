"""Explicit completion records for reusable derived outputs."""
import hashlib
import json
from pathlib import Path


def signature(parameters, inputs):
    # Stat identity avoids reading multi-gigabyte sensor files a second time.
    files = []
    for item in inputs:
        if item is None:
            files.append(None)
        else:
            p = Path(item).resolve()
            stat = p.stat() if p.exists() else None
            files.append((str(p), None if stat is None else (stat.st_size, stat.st_mtime_ns)))
    encoded = json.dumps({"revision": 1, "parameters": parameters, "inputs": files}, default=str, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OutputRun:
    def __init__(self, marker, outputs):
        self.marker = Path(marker)
        self.outputs = [Path(p) for p in outputs]

    def reusable(self, key):
        try:
            record = json.loads(self.marker.read_text())
            return record["signature"] == key and record["outputs"] == {str(p.name): _digest(p) for p in self.outputs}
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def clear(self):
        self.marker.unlink(missing_ok=True)
        for path in self.outputs:
            path.unlink(missing_ok=True)

    def complete(self, key):
        record = {"signature": key, "outputs": {str(p.name): _digest(p) for p in self.outputs}}
        temp = self.marker.with_suffix(".tmp")
        temp.write_text(json.dumps(record, sort_keys=True))
        temp.replace(self.marker)


def fresh_outputs(patterns, *, folder_argument="save_folder"):
    """Invalidate owned derived tables before a rerun and remove partial failures."""
    from functools import wraps
    from inspect import signature as function_signature
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = function_signature(function).bind(*args, **kwargs)
            folder = bound.arguments.get(folder_argument)
            def clear():
                if folder is not None:
                    for pattern in patterns:
                        for path in Path(folder).glob(pattern):
                            path.unlink()
            clear()
            try:
                return function(*args, **kwargs)
            except Exception:
                clear()
                raise
        return wrapped
    return decorate


def clear_outputs_on_failure(output_spec):
    """Remove derived outputs if an input read, computation or write fails.

    The spec maps a subdirectory argument to that stage's owned filenames.
    Successful reusable runs are left intact.
    """
    from functools import wraps
    from inspect import signature as function_signature
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = function_signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            options = bound.arguments
            root = Path(options["output_root"]) / str(options["participant"]) / str(options["visit"]) / str(options["sensor"])
            try:
                return function(*args, **kwargs)
            except Exception:
                for argument, filenames in output_spec.items():
                    for filename in filenames:
                        (root / options[argument] / filename).unlink(missing_ok=True)
                raise
        return wrapped
    return decorate
