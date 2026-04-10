from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from benchmark_AD.data import LabeledSample, list_labeled_images, normalize_0_1, read_image_bgr, resize

from .contracts import FramePacket


@dataclass
class FolderInputHandler:
    root_dir: Path
    resize_wh: Optional[tuple[int, int]] = None
    normalize: bool = True
    loop: bool = False
    max_frames: Optional[int] = None
    sequence_mode: str = "interleaved_labels"

    def _samples(self) -> list[LabeledSample]:
        samples = list_labeled_images(self.root_dir)
        if str(self.sequence_mode).lower() != "interleaved_labels":
            return samples
        return _interleave_by_label(samples)

    def iter_frames(self) -> Iterator[FramePacket]:
        samples = self._samples()
        if len(samples) == 0:
            raise ValueError(f"No image files found in runtime input folder: {self.root_dir}")

        emitted = 0
        frame_id = 0

        while True:
            for sample in samples:
                if self.max_frames is not None and emitted >= self.max_frames:
                    return

                raw_image = read_image_bgr(str(sample.path))
                if self.resize_wh is not None:
                    raw_image = resize(raw_image, self.resize_wh)

                model_input = normalize_0_1(raw_image) if self.normalize else raw_image
                yield FramePacket(
                    frame_id=frame_id,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    path=Path(sample.path),
                    raw_image_bgr=raw_image,
                    model_input=model_input,
                    label=sample.label,
                    defect_type=sample.defect_type,
                )
                frame_id += 1
                emitted += 1

            if not self.loop:
                return


def _interleave_by_label(samples: list[LabeledSample]) -> list[LabeledSample]:
    buckets = {
        0: deque([sample for sample in samples if sample.label == 0]),
        1: deque([sample for sample in samples if sample.label == 1]),
        -1: deque([sample for sample in samples if sample.label == -1]),
    }

    ordered: list[LabeledSample] = []
    label_cycle = [0, 1, -1]
    while any(buckets[label] for label in label_cycle):
        for label in label_cycle:
            if buckets[label]:
                ordered.append(buckets[label].popleft())
    return ordered
