# OTB100 Classical Object Tracking Playground

**[中文文档](./README.zh-CN.md)** | English

A small, from-scratch exploration of **classical (non-deep-learning) single-object
tracking** methods, evaluated on sequences from the [OTB100](http://cvlab.hanyang.ac.kr/tracker_benchmark/)
benchmark. This project was built as a learning exercise to understand how far
"traditional" computer vision techniques (background subtraction, frame
differencing, color histograms, edge templates) can go before deep
learning-based trackers became the default.

Four tracking pipelines are implemented, each following the same
`Tracker + Scorer` structure so they can be compared fairly:

| Tracker | File | Core idea |
|---|---|---|
| Frame Difference | `frame_diff_tracker.py` | Absolute difference between consecutive frames |
| Background Difference | `background_diff_tracker.py` | Absolute difference against a slowly-updated background model |
| Color Histogram | `color_thresh_tracker.py` | HSV back-projection of the target's color histogram (CamShift-style) |
| Edge Template | `edge_detect_tracker.py` | Canny edge map + local template matching within a search window |

None of these methods is a state-of-the-art tracker — they're intentionally
simple, interpretable, and easy to break, which makes them a good way to
*feel* the failure modes (occlusion, drift, template pollution, illumination
change) that motivated more modern tracking approaches.

## ⚠️ Status / honesty note

This is exploratory / coursework-style code, not a polished library. A few
things worth knowing before you dig in:

- The tracker classes share a lot of structure (mask generation → candidate
  scoring → template update) but are **not yet refactored into a common base
  class**. Contributions that reduce duplication are welcome.
- Template updating (adapting the appearance model frame-to-frame) is
  implemented but, empirically, is **not always beneficial** — on several
  sequences a *frozen* initial template outperforms an adaptively-updated one,
  because a single bad match can "poison" the template and the tracker never
  recovers. This tradeoff is discussed more in [Notes on template updating](#notes-on-template-updating)
  below.
- These trackers do not handle scale change, rotation, or long-term full
  occlusion well. They're meant for learning/comparison, not production use.

## Project structure

```
.
├── background_diff_tracker.py   # background-subtraction tracker + demo script
├── color_thresh_tracker.py      # HSV histogram back-projection tracker + demo script
├── edge_detect_tracker.py       # Canny edge template tracker + demo script
├── frame_diff_tracker.py        # frame-differencing tracker + demo script
└── utils/
    ├── loader.py                 # threaded OTB100 sequence/frame loader
    ├── selectors.py               # "Scorer" classes: turn candidate boxes into a final prediction
    └── statistic.py               # IoU / CLE / success-rate evaluation
```

### How it fits together

1. **`OTB100Loader`** (`utils/loader.py`) streams `(frame, ground_truth_box)`
   pairs from an OTB100 sequence folder on a background thread, so frame
   decoding doesn't block the tracking loop.
2. Each **Tracker** class (e.g. `EdgeDetectTracker`) turns the current frame
   into a binary "mask" using its own method (edges, color back-projection,
   frame diff, etc.), then hands that mask to a **Scorer**.
3. Each **Scorer** class (in `utils/selectors.py`) turns a mask into a single
   predicted bounding box, by finding candidate regions and ranking them with
   a combination of *appearance similarity* (to a stored template) and
   *motion continuity* (distance to the previous prediction).
4. **`Statistic`** (`utils/statistic.py`) accumulates IoU and Center Location
   Error (CLE) per frame and prints summary statistics (mean/median/std,
   best/worst frame, success rate at a given IoU threshold) at the end of a
   run.

Bounding boxes are represented as `(x, y, w, h)` throughout the project
(top-left corner + width/height), matching the OTB100 ground-truth format.

## Installation

```bash
git clone https://github.com/Cco-fu/classical-object-tracking.git
cd classical-object-tracking
pip install -r requirements.txt
```

## Getting the data

This project expects the [OTB100 dataset](http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html)
to be available locally, with each sequence laid out as:

```
OTB100/
└── Walking/
    ├── img/
    │   ├── 0001.jpg
    │   ├── 0002.jpg
    │   └── ...
    └── groundtruth_rect.txt
```

By default the scripts look for the dataset at `../OTB100` relative to this
project's folder — adjust the `OTB100Loader(...)` path argument at the bottom
of each tracker script (or the `DATA_NAME` constant) to point at your local
copy and the sequence you want to test.

## Running a tracker

Each tracker file is also a runnable demo script. For example:

```bash
python edge_detect_tracker.py
```

This will:
- open two OpenCV windows showing the tracked box vs. ground truth, and the
  intermediate mask,
- print FPS and summary statistics (mean IoU, mean CLE, success rate) at the
  end of the sequence.

Press `Esc` in the video window to stop early.

Key parameters (kernel size, binarization threshold, template update rate,
search radius, etc.) are exposed via a `parameters` dict near the bottom of
each script — see the docstring of each `Tracker` class for what each one
does.

### Debug mode

Every tracker has a `Debug*` subclass (e.g. `DebugEdgeDetectTracker`) that
writes intermediate images (edge maps, masks, contours, per-frame results) to
disk under `./output/<sequence>/frame<N>/`, useful for visually diagnosing
why a tracker is failing on a particular sequence. Toggle this with the
`DEBUG` flag at the top of each script.

## Notes on template updating

A recurring theme in this project: **adaptive template updates are a
double-edged sword.**

- With `template_alpha = 0` (no update), the tracker always compares against
  the original, hand-labeled first frame. This is robust to short occlusions
  and background clutter, but can't adapt to genuine appearance changes
  (lighting, pose, scale).
- With `template_alpha > 0`, the template is blended each frame
  (`template = (1-α)·template + α·new_patch`). This can look "smoother" when
  it works, but if a single frame is mismatched (occlusion, background edge
  clutter, etc.), that error gets baked into the template — and because the
  update is a positive feedback loop, the tracker can drift permanently with
  no way to self-correct.

A few mitigations worth exploring (some partially prototyped, none fully
settled as "best"):
- only update the template when the match score is confidently above
  threshold (avoid updating on marginal matches),
- maintain both a frozen initial template and an adaptive one, and combine
  their scores,
- use a rolling window of recently-accepted patches and only accept a new
  update if it's consistent with the *median* of recent history, rather than
  trusting any single "ground truth" template.

If you try one of these and get solid results on a specific sequence, a PR
with findings (and ideally before/after IoU numbers) would be very welcome.

## Evaluation metrics

- **IoU** (Intersection over Union) between predicted and ground-truth boxes.
- **CLE** (Center Location Error) — Euclidean distance between predicted and
  ground-truth box centers, in pixels.
- **Success rate** — fraction of frames with IoU above a configurable
  threshold (default 0.5).

## Known limitations

- No scale adaptation — predicted box size is fixed to the template size (or
  bounding-rect of a contour), so trackers struggle when the target grows or
  shrinks significantly on screen.
- No re-detection after long occlusion beyond simple "expand search radius
  after N lost frames" logic.
- Color/edge features are fairly weak compared to modern learned features,
  so performance is very sequence-dependent (works well on some OTB100
  sequences, poorly on others with heavy clutter or occlusion).
- Not benchmarked against the full OTB100 success/precision plots — only
  simple per-sequence summary stats are printed.

## License

MIT — see [LICENSE](./LICENSE).

## Contributing

This started as a personal learning project, so the code style and structure
still show that (some duplication between tracker classes, some
work-in-progress experiments left in as commented-out code). Issues and PRs
around refactoring, bug fixes, or extending the evaluation (e.g. full
OTB100 success-plot AUC) are welcome.
