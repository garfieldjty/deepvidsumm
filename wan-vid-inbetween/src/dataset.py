# src/dataset.py

from typing import List, Dict, Optional
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------
# Video Loading Helpers
# ----------------------------

def read_video_frames(path: str, num_frames: int, start_index: Optional[int] = None) -> List[np.ndarray]:
    """
    Read num_frames RGB frames starting from start_index.
    Pads by repeating the last frame if not enough frames available.
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError(f"Video {path} has zero frames.")

    # Determine start index
    if start_index is None:
        if total <= num_frames:
            start_index = 0
        else:
            start_index = np.random.randint(0, total - num_frames + 1)
    else:
        start_index = max(0, min(start_index, total - 1))

    end_index = start_index + num_frames

    frames = []
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id >= start_index and frame_id < end_index:
            frames.append(frame[..., ::-1])  # BGR -> RGB

        frame_id += 1
        if frame_id >= end_index:
            break

    cap.release()

    # Pad if fewer frames than needed
    if len(frames) < num_frames:
        last = frames[-1]
        frames += [last] * (num_frames - len(frames))

    return frames


def resize_frames(frames: List[np.ndarray], height: int, width: int) -> List[np.ndarray]:
    # Batch resize for better performance
    if not frames:
        return []
    # Use INTER_LINEAR for faster resize (INTER_AREA is slower but higher quality)
    out = []
    for f in frames:
        out.append(cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR))
    return out


# ----------------------------
# Dataset with Cut-Focused Sampling
# ----------------------------

class InbetweenVideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        video_glob: str,
        clip_num_frames: int,
        height: int,
        width: int,
        inbetween_mode: str = "30_60_30",
        start_frames: int = 30,
        mid_frames: int = 60,
        end_frames: int = 30,
        cut_annotations_path: Optional[str] = None,
        use_cut_focused_sampling: bool = True,
        cut_focus_prob: float = 0.7,
    ):
        self.data_root = Path(data_root)
        self.paths = sorted(self.data_root.glob(video_glob))
        if not self.paths:
            raise ValueError(f"No videos found in {data_root} with pattern {video_glob}")

        self.clip_num_frames = clip_num_frames
        self.height = height
        self.width = width

        self.s = start_frames
        self.m = mid_frames
        self.e = end_frames
        assert (self.s + self.m + self.e) == clip_num_frames, "start+mid+end must equal clip_num_frames"

        # Cut sampling control
        self.use_cut_focused_sampling = use_cut_focused_sampling
        self.cut_focus_prob = cut_focus_prob

        # Load cut metadata
        self.cut_annotations: Dict[str, List[int]] = {}
        if cut_annotations_path:
            with open(cut_annotations_path, "r") as f:
                raw = json.load(f)
            # keys are video filenames, values are sorted list of cut frame indices
            self.cut_annotations = {k: sorted(v) for k, v in raw.items()}

    # ----------------------------
    # Helpers
    # ----------------------------
    def _get_video_key(self, path: Path) -> str:
        # Match annotation keys by filename
        return path.name

    def _sample_clip_round_cut(self, path: Path) -> Optional[List[np.ndarray]]:
        """
        Extract a clip such that a known hard cut lies inside the middle region.
        """
        key = self._get_video_key(path)
        cuts = self.cut_annotations.get(key, [])
        if not cuts:
            return None  # fallback
        
        # Pick a random cut index
        c = int(np.random.choice(cuts))

        # We want the cut in the center of the mid region:
        mid_center = self.s + self.m // 2  # for 30+60+30 = 30+30=60

        # Start window so that c aligns with mid_center
        start_idx = c - mid_center
        start_idx = max(0, start_idx)

        frames = read_video_frames(str(path), self.clip_num_frames, start_index=start_idx)
        frames = resize_frames(frames, self.height, self.width)
        return frames

    def _sample_random_clip(self, path: Path) -> List[np.ndarray]:
        frames = read_video_frames(str(path), self.clip_num_frames)
        frames = resize_frames(frames, self.height, self.width)
        return frames

    # ----------------------------
    # Main Access
    # ----------------------------
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        vid_path = self.paths[idx]

        # Decide sampling strategy
        use_cut = (
            self.use_cut_focused_sampling and
            (self.cut_annotations is not None) and
            (self._get_video_key(vid_path) in self.cut_annotations) and
            (np.random.rand() < self.cut_focus_prob)
        )

        if use_cut:
            frames = self._sample_clip_round_cut(vid_path)
            if frames is None:
                frames = self._sample_random_clip(vid_path)
        else:
            frames = self._sample_random_clip(vid_path)

        # Convert to tensor [T, C, H, W]
        video_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_np = np.transpose(video_np, (0, 3, 1, 2))
        video = torch.from_numpy(video_np)

        return {
            "video": video,    # [T, 3, H, W]
            "path": str(vid_path),
        }
