"""RadioML and HisarMod loading, splitting, and cache preparation."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import hashlib
import pickle
import random
import argparse
import gc
import os
from tqdm import tqdm

SIGNAL_LENGTH = 1024
NORMALIZATION_EPSILON = 1e-12
HISARMOD_SNRS = tuple(range(-20, 20, 2))
RAW_LABEL_CODES = (
    0,
    1,
    2,
    3,
    4,
    10,
    11,
    12,
    13,
    14,
    20,
    21,
    22,
    23,
    24,
    30,
    31,
    32,
    34,
    40,
    41,
    44,
    50,
    51,
    54,
    61,
)
HISARMOD_MODULATIONS = (
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "64PSK",
    "4QAM",
    "8QAM",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "2FSK",
    "4FSK",
    "8FSK",
    "16FSK",
    "4PAM",
    "8PAM",
    "16PAM",
    "AM-DSB",
    "AM-DSB-SC",
    "AM-USB",
    "AM-LSB",
    "FM",
    "PM",
)
OFFICIAL_TRAIN_PER_CLASS_SNR = 1000
OFFICIAL_TEST_PER_CLASS_SNR = 500
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data" / "hisarmod2019" / "cache"


def split_files(data_dir: str | Path, split: str) -> dict[str, Path]:
    split_name = split.capitalize()
    prefix = split.lower()
    data_dir = Path(data_dir)
    flat = {
        "data": data_dir / f"{prefix}_data.csv",
        "labels": data_dir / f"{prefix}_labels.csv",
        "snr": data_dir / f"{prefix}_snr.csv",
    }
    if all((path.is_file() for path in flat.values())):
        return flat
    nested_dir = data_dir / split_name
    return {
        "data": nested_dir / f"{prefix}_data.csv",
        "labels": nested_dir / f"{prefix}_labels.csv",
        "snr": nested_dir / f"{prefix}_snr.csv",
    }


def cache_paths(cache_dir: str | Path, split: str) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    prefix = split.lower()
    return (cache_dir / f"{prefix}_signals.npy", cache_dir / f"{prefix}_cache.json")


def load_metadata(data_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    files = split_files(data_dir, split)
    labels = np.loadtxt(files["labels"], dtype=np.int64, ndmin=1)
    snrs = np.loadtxt(files["snr"], dtype=np.int64, ndmin=1)
    if labels.shape != snrs.shape:
        raise ValueError(
            f"{split}: labels and SNR lengths differ: {labels.shape} vs {snrs.shape}"
        )
    return (labels, snrs)


def _expected_count(split: str) -> int:
    samples = (
        OFFICIAL_TRAIN_PER_CLASS_SNR
        if split.lower() == "train"
        else OFFICIAL_TEST_PER_CLASS_SNR
    )
    return len(HISARMOD_MODULATIONS) * len(HISARMOD_SNRS) * samples


def validate_raw_layout(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    label_set = set(RAW_LABEL_CODES)
    snr_set = set(HISARMOD_SNRS)
    for split in ("train", "test"):
        files = split_files(data_dir, split)
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing HisarMod CSV file(s): " + ", ".join(missing)
            )
        labels, snrs = load_metadata(data_dir, split)
        expected = _expected_count(split)
        if labels.size != expected:
            raise ValueError(
                f"{split}: expected {expected} samples, found {labels.size}"
            )
        if set(labels.tolist()) != label_set:
            raise ValueError(
                f"{split}: unexpected raw labels {sorted(set(labels.tolist()))}"
            )
        if set(snrs.tolist()) != snr_set:
            raise ValueError(f"{split}: unexpected SNRs {sorted(set(snrs.tolist()))}")
        expected_per_cell = (
            OFFICIAL_TRAIN_PER_CLASS_SNR
            if split == "train"
            else OFFICIAL_TEST_PER_CLASS_SNR
        )
        cell_keys = labels * 100 + (snrs + 20)
        unique_keys, counts = np.unique(cell_keys, return_counts=True)
        count_by_key = dict(zip(unique_keys.tolist(), counts.tolist(), strict=True))
        for raw_label in RAW_LABEL_CODES:
            for snr in HISARMOD_SNRS:
                count = int(count_by_key.get(raw_label * 100 + (snr + 20), 0))
                if count != expected_per_cell:
                    raise ValueError(
                        f"{split}: label={raw_label}, SNR={snr} has {count} samples; expected {expected_per_cell}"
                    )
        with files["data"].open("rb") as handle:
            first = np.fromstring(
                handle.readline().replace(b"i", b"j"), dtype=np.complex64, sep=","
            )
        if first.size != SIGNAL_LENGTH:
            raise ValueError(
                f"{split}: expected {SIGNAL_LENGTH} complex values per row, got {first.size}"
            )


def _source_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def cache_is_current(data_dir: str | Path, cache_dir: str | Path, split: str) -> bool:
    files = split_files(data_dir, split)
    signal_path, manifest_path = cache_paths(cache_dir, split)
    if not signal_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signals = np.load(signal_path, mmap_mode="r")
        valid = (
            tuple(signals.shape) == (_expected_count(split), 2, SIGNAL_LENGTH)
            and signals.dtype in (np.dtype("float32"), np.dtype("float16"))
            and (int(manifest["source"]["size"]) == int(files["data"].stat().st_size))
            and (manifest["shape"] == list(signals.shape))
            and (manifest["dtype"] == str(signals.dtype))
        )
        del signals
        return bool(valid)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def require_cache(
    data_dir: str | Path, cache_dir: str | Path = DEFAULT_CACHE_DIR
) -> None:
    missing = [
        split
        for split in ("train", "test")
        if not cache_is_current(data_dir, cache_dir, split)
    ]
    if missing:
        command = f'python dataset.py --prepare --data-path "{Path(data_dir)}" --cache-dir "{Path(cache_dir)}"'
        raise FileNotFoundError(
            f"Missing or stale binary cache for {', '.join(missing)}. Run once from this folder: {command}"
        )


def encode_labels(raw_labels: np.ndarray) -> np.ndarray:
    lookup = {raw: index for index, raw in enumerate(RAW_LABEL_CODES)}
    try:
        return np.fromiter(
            (lookup[int(value)] for value in raw_labels),
            dtype=np.int64,
            count=raw_labels.size,
        )
    except KeyError as error:
        raise ValueError(f"Unknown HisarMod raw label: {error.args[0]}") from error


def normalize_unit_complex_power(
    signal: np.ndarray, epsilon: float = NORMALIZATION_EPSILON
) -> np.ndarray:
    """Normalize one [2, L] I/Q sample to unit average complex power."""
    if signal.shape != (2, SIGNAL_LENGTH):
        raise ValueError(
            f"Expected one I/Q sample with shape (2, {SIGNAL_LENGTH}), got {signal.shape}"
        )
    if epsilon <= 0:
        raise ValueError("Normalization epsilon must be positive")
    in_phase = signal[0].astype(np.float64, copy=False)
    quadrature = signal[1].astype(np.float64, copy=False)
    power = float(np.mean(in_phase * in_phase + quadrature * quadrature))
    if not np.isfinite(power):
        raise ValueError("HisarMod sample contains a non-finite I/Q value")
    signal /= np.float32(np.sqrt(max(power, epsilon)))
    return signal


def official_8_2_5_indices(
    raw_train_labels: np.ndarray, train_snrs: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the official 1000-sample Train cells into 800 train + 200 val."""
    generator = np.random.default_rng(seed)
    train_parts, validation_parts = ([], [])
    for raw_label in RAW_LABEL_CODES:
        for snr in HISARMOD_SNRS:
            indices = np.flatnonzero(
                (raw_train_labels == raw_label) & (train_snrs == snr)
            )
            if indices.size != OFFICIAL_TRAIN_PER_CLASS_SNR:
                raise ValueError(
                    f"Expected 1000 Train samples for label={raw_label}, SNR={snr}; found {indices.size}"
                )
            indices = generator.permutation(indices)
            train_parts.append(indices[:800])
            validation_parts.append(indices[800:])
    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    return (train_indices, validation_indices)


class HisarModDataset(Dataset):
    """Worker-safe lazy view of an on-disk NPY signal cache."""

    def __init__(self, signal_path, indices, labels, snrs) -> None:
        self.signal_path = str(Path(signal_path).resolve())
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.snrs = np.asarray(snrs, dtype=np.int64)
        self._signals = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _ensure_open(self) -> None:
        if self._signals is None:
            self._signals = np.load(self.signal_path, mmap_mode="r")

    def __getitem__(self, item):
        self._ensure_open()
        source_index = int(self.indices[item])
        signal = np.array(self._signals[source_index], dtype=np.float32, copy=True)
        signal = normalize_unit_complex_power(signal)
        return (
            torch.from_numpy(signal),
            torch.tensor(self.labels[source_index], dtype=torch.long),
            torch.tensor(self.snrs[source_index], dtype=torch.long),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_signals"] = None
        return state


def get_hisarmod_loaders(
    data_path,
    batch_size=128,
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.5,
    seed=42,
    mods=None,
    snrs=None,
    num_workers=8,
    cache_dir=DEFAULT_CACHE_DIR,
    eval_batch_size=None,
):
    if (train_ratio, val_ratio, test_ratio) != (0.8, 0.2, 0.5):
        raise ValueError(
            "HisarMod benchmark protocol is fixed at 8:2:5 (800/200/500 per cell)"
        )
    if mods is not None or snrs is not None:
        raise ValueError("This benchmark uses all 26 classes and all 20 SNR levels")
    validate_raw_layout(data_path)
    require_cache(data_path, cache_dir)
    raw_train_labels, train_snrs = load_metadata(data_path, "train")
    raw_test_labels, test_snrs = load_metadata(data_path, "test")
    train_labels = encode_labels(raw_train_labels)
    test_labels = encode_labels(raw_test_labels)
    train_indices, validation_indices = official_8_2_5_indices(
        raw_train_labels, train_snrs, seed
    )
    test_indices = np.arange(test_labels.size, dtype=np.int64)
    train_signal_path, _ = cache_paths(cache_dir, "train")
    test_signal_path, _ = cache_paths(cache_dir, "test")
    train_dataset = HisarModDataset(
        train_signal_path, train_indices, train_labels, train_snrs
    )
    validation_dataset = HisarModDataset(
        train_signal_path, validation_indices, train_labels, train_snrs
    )
    test_dataset = HisarModDataset(
        test_signal_path, test_indices, test_labels, test_snrs
    )
    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    evaluation_batch_size = (
        batch_size if eval_batch_size is None else int(eval_batch_size)
    )
    if evaluation_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    generator = torch.Generator().manual_seed(int(seed) + 10000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **common_loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    metadata = {
        "dataset": "HisarMod2019.1",
        "mods": list(HISARMOD_MODULATIONS),
        "label2mod": dict(enumerate(HISARMOD_MODULATIONS)),
        "raw_label_codes": list(RAW_LABEL_CODES),
        "snrs": list(HISARMOD_SNRS),
        "num_classes": len(HISARMOD_MODULATIONS),
        "signal_length": SIGNAL_LENGTH,
        "input_normalization": "per_sample_unit_average_complex_power",
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "split_protocol": "8:2:5 (800 train, 200 validation, 500 test per class/SNR)",
        "train_size": len(train_dataset),
        "val_size": len(validation_dataset),
        "test_size": len(test_dataset),
        "train_batch_size": int(batch_size),
        "eval_batch_size": evaluation_batch_size,
    }
    return (train_loader, validation_loader, test_loader, metadata)


DATASETS = ("rml2016a", "rml2016b", "rml2022", "hisarmod2019")
DEFAULT_PATHS = {
    "rml2016a": "data/RML2016.10a_dict.pkl",
    "rml2016b": "data/RML2016.10b.dat",
    "rml2022": "data/RML22.01A",
    "hisarmod2019": "data/hisarmod2019",
}
BASE_DIR = Path(__file__).resolve().parent


def default_data_path(dataset: str) -> Path:
    return BASE_DIR / DEFAULT_PATHS[dataset]


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def stratified_split(
    labels, snrs, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
):
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if (
        not np.isfinite(ratios).all()
        or np.any(ratios <= 0)
        or (not np.isclose(ratios.sum(), 1))
    ):
        raise ValueError("Split ratios must be positive and sum to one")
    labels, snrs = (np.asarray(labels), np.asarray(snrs))
    if labels.ndim != 1 or labels.shape != snrs.shape or (not labels.size):
        raise ValueError("labels and snrs must be nonempty, equally sized vectors")
    generator = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in np.unique(labels):
        for snr in np.unique(snrs):
            indices = np.flatnonzero((labels == label) & (snrs == snr))
            if not indices.size:
                continue
            generator.shuffle(indices)
            train_end = int(round(indices.size * train_ratio))
            val_end = train_end + int(round(indices.size * val_ratio))
            if train_end == 0 or val_end <= train_end or val_end >= indices.size:
                raise ValueError(
                    f"Too few samples for three splits: class={label}, SNR={snr}"
                )
            for output, section in zip(
                parts,
                (indices[:train_end], indices[train_end:val_end], indices[val_end:]),
                strict=True,
            ):
                output.append(section)
    result = tuple((np.concatenate(part) for part in parts))
    for indices in result:
        generator.shuffle(indices)
    return result


class RadioMLDataset(Dataset):
    """Indexed view of pickle blocks without copying the signal array per split."""

    def __init__(self, blocks, offsets, labels, snrs, indices, mods):
        self.blocks = blocks
        self.offsets = offsets
        self.labels = labels
        self.snrs = snrs
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mods = list(mods)

    def __len__(self):
        return self.indices.size

    def __getitem__(self, item):
        index = int(self.indices[item])
        block = int(np.searchsorted(self.offsets, index, side="right") - 1)
        signal = np.array(
            self.blocks[block][index - self.offsets[block]], dtype=np.float32, copy=True
        )
        if not np.isfinite(signal).all():
            raise ValueError(f"Non-finite I/Q sample at index {index}")
        return (
            torch.from_numpy(signal),
            int(self.labels[index]),
            int(self.snrs[index]),
        )


def load_radioml(path: str | Path, expected_length: int):
    with Path(path).open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    if not isinstance(data, dict) or not data:
        raise ValueError("Expected a nonempty dictionary keyed by (modulation, SNR)")
    mods = sorted({key[0] for key in data})
    snrs = sorted({int(key[1]) for key in data})
    blocks, labels, sample_snrs, offsets = ([], [], [], [0])
    for label, modulation in enumerate(mods):
        for snr in snrs:
            block = data.get((modulation, snr))
            if block is None:
                continue
            block = np.asarray(block)
            if block.ndim != 3 or block.shape[1:] != (2, expected_length):
                raise ValueError(
                    f"{modulation}, {snr}: expected [N, 2, {expected_length}], got {block.shape}"
                )
            if not np.issubdtype(block.dtype, np.floating):
                raise ValueError("I/Q blocks must contain real floating-point values")
            if not len(block):
                raise ValueError(f"Empty block: {modulation}, {snr}")
            blocks.append(block)
            labels.append(np.full(len(block), label, dtype=np.int64))
            sample_snrs.append(np.full(len(block), snr, dtype=np.int64))
            offsets.append(offsets[-1] + len(block))
    return (
        blocks,
        np.asarray(offsets),
        np.concatenate(labels),
        np.concatenate(sample_snrs),
        mods,
    )


def split_digest(datasets) -> str:
    digest = hashlib.sha256()
    for split in datasets:
        digest.update(np.asarray(split.indices, dtype="<i8").tobytes())
        digest.update(np.asarray(split.labels, dtype="<i8").tobytes())
        digest.update(np.asarray(split.snrs, dtype="<i8").tobytes())
    return digest.hexdigest()


def get_data_loaders(
    data_path=None,
    batch_size=128,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=3407,
    num_workers=0,
    eval_batch_size=400,
    dataset="rml2016a",
    cache_dir=None,
):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if batch_size < 1 or eval_batch_size < 1 or num_workers < 0:
        raise ValueError(
            "Batch sizes must be positive and num_workers must be nonnegative"
        )
    path = Path(data_path) if data_path is not None else default_data_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Supply --data-path or see README.md."
        )
    if dataset == "hisarmod2019":
        if (train_ratio, val_ratio, test_ratio) != (0.6, 0.2, 0.2):
            raise ValueError("HisarMod uses its fixed official 800/200/500 split")
        cache = Path(cache_dir) if cache_dir is not None else path / "cache"
        train, val, test, metadata = get_hisarmod_loaders(
            path,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            cache_dir=cache,
            eval_batch_size=eval_batch_size,
        )
        loaders = (train, val, test)
    else:
        length = 128
        blocks, offsets, labels, snrs, mods = load_radioml(path, length)
        indices = stratified_split(
            labels, snrs, train_ratio, val_ratio, test_ratio, seed
        )
        splits = [
            RadioMLDataset(blocks, offsets, labels, snrs, ids, mods) for ids in indices
        ]
        generator = torch.Generator().manual_seed(seed + 10000)
        common = dict(
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
        )
        loaders = (
            DataLoader(
                splits[0],
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                **common,
            ),
            DataLoader(splits[1], batch_size=eval_batch_size, shuffle=False, **common),
            DataLoader(splits[2], batch_size=eval_batch_size, shuffle=False, **common),
        )
        metadata = {
            "mods": mods,
            "label2mod": dict(enumerate(mods)),
            "snrs": sorted(np.unique(snrs).tolist()),
            "num_classes": len(mods),
            "signal_length": length,
            "input_normalization": "none",
            "split_protocol": [train_ratio, val_ratio, test_ratio],
        }
    metadata.pop("train_batch_size", None)
    metadata.pop("eval_batch_size", None)
    metadata.update(
        dataset=dataset,
        seed=seed,
        train_size=len(loaders[0].dataset),
        val_size=len(loaders[1].dataset),
        test_size=len(loaders[2].dataset),
        train_loader_seed=seed + 10000,
        split_digest=split_digest([loader.dataset for loader in loaders]),
    )
    return (*loaders, metadata)


def prepare_split(data_dir: Path, cache_dir: Path, split: str, dtype: str, force: bool):
    source_path = split_files(data_dir, split)["data"]
    output_path, manifest_path = cache_paths(cache_dir, split)
    if not force and cache_is_current(data_dir, cache_dir, split):
        print(f"{split}: current cache already exists: {output_path}", flush=True)
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    expected = _expected_count(split)
    output = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=(expected, 2, SIGNAL_LENGTH),
    )
    try:
        with source_path.open("rb", buffering=8 * 1024 * 1024) as source:
            progress = tqdm(
                source,
                total=expected,
                desc=f"Converting {split}",
                unit="sample",
                mininterval=1.0,
            )
            count = 0
            for count, line in enumerate(progress, start=1):
                values = np.fromstring(
                    line.replace(b"i", b"j"), dtype=np.complex64, sep=","
                )
                if values.size != SIGNAL_LENGTH:
                    raise ValueError(
                        f"{source_path}: row {count} contains {values.size} values; expected {SIGNAL_LENGTH}"
                    )
                if count > expected:
                    raise ValueError(f"{source_path}: more than {expected} rows")
                if not np.isfinite(values).all():
                    raise ValueError(f"{source_path}: non-finite value in row {count}")
                output[count - 1, 0] = values.real
                output[count - 1, 1] = values.imag
                if not np.isfinite(output[count - 1]).all():
                    raise ValueError(
                        f"{source_path}: row {count} overflows {dtype}; use float32"
                    )
            if count != expected:
                raise ValueError(
                    f"{source_path}: expected {expected} rows, found {count}"
                )
        output.flush()
        del output
        output = None
        gc.collect()
        os.replace(temporary_path, output_path)
        manifest = {
            "dataset": "HisarMod2019.1",
            "split": split,
            "source": _source_fingerprint(source_path),
            "shape": [expected, 2, SIGNAL_LENGTH],
            "dtype": dtype,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{split}: wrote {output_path}", flush=True)
    except BaseException:
        if output is not None:
            del output
        gc.collect()
        if temporary_path.exists():
            temporary_path.unlink()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare", action="store_true", help="Convert HisarMod CSV signals to NPY"
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if not args.prepare:
        parser.error("Use --prepare to build the HisarMod signal cache")
    validate_raw_layout(args.data_path)
    cache = args.cache_dir or args.data_path / "cache"
    for split in ("train", "test"):
        prepare_split(args.data_path, cache, split, "float32", False)
