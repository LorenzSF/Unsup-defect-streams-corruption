# BEST_PRACTICES.md - Code-change rules for GenAI agents

This file is the checklist for any AI or human agent modifying this repository.
Its purpose is to prevent "vibe coding": large, plausible-looking diffs that add
surface area, hidden assumptions, unused abstractions, and technical debt.

If a requested change conflicts with these rules, stop and explain the conflict.
Do not weaken the rules to make a diff easier.

---

## 0. Project goals

Every line of code must serve at least one of these goals:

1. **Streaming inference and visualization.** Process industrial image streams
   and produce explainable anomaly outputs.
2. **Standardized benchmarking.** Run SOTA unsupervised IAD detectors under the
   same stream, corruption, threshold, and seed settings, then compare their
   `report.json` outputs.
3. **Robustness under environmental noise.** Measure the effect of synthetic
   corruptions such as noise, blur, brightness shifts, and contrast loss.

If code does not support one of these goals, remove it or do not add it.

---

## 1. Anti-vibe-coding principles

1. **Smallest useful diff.** Make the narrowest change that satisfies the task.
   Do not perform drive-by refactors, renames, formatting sweeps, or architecture
   rewrites.
2. **Read before editing.** Inspect the relevant modules and existing patterns
   before changing code. Do not infer architecture from filenames alone.
3. **No speculative features.** Do not add options, modes, fallback behavior,
   dashboards, abstractions, or extension points unless the task requires them.
4. **No fake completeness.** Do not add TODO-heavy scaffolding, placeholder
   classes, unreachable branches, or "future-proof" APIs.
5. **Delete dead code.** Remove unused functions, imports, variables, comments,
   and branches introduced by the change.
6. **Prefer simple code over clever code.** A direct typed function is better
   than a generic framework when only one use case exists.
7. **Ask only on blocking ambiguity.** If ambiguity affects behavior, data
   semantics, architecture, or evaluation validity, ask before coding. Otherwise
   make the conservative choice that matches the existing code.
8. **Do not hide uncertainty.** If a change depends on an unverified assumption,
   either verify it locally or state it clearly.
9. **Code first, explanation second.** The final result should be working code,
   not a long explanation around weak or incomplete code.

---

## 2. Architectural invariants

These are hard constraints for this repository:

1. **One execution path.** [main.py](main.py) is the only orchestrator. Do not
   add `train.py`, `evaluate.py`, `quick_test.py`, notebooks-as-entry-points, or
   alternate runners.
2. **One config surface.** Runtime settings live in [config.yaml](config.yaml).
   Do not add CLI flags, scattered environment-variable knobs, or ad hoc module
   globals.
3. **Mandatory warm-up.** The run always performs warm-up between `build_model`
   and the streaming loop. It is not optional.
4. **One dependency file.** Dependencies stay in [requirements.txt](requirements.txt).
   Do not add Poetry, Conda, `pyproject.toml`, or extra requirements files unless
   explicitly requested.
5. **Limited documentation files.** Top-level Markdown docs are limited to
   [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), and this file.
6. **Flat source layout.** Keep the current flat `src/` modules. Do not create
   per-model packages, `utils.py`, `helpers.py`, `common.py`, or `misc.py`.
7. **Dataclasses across module boundaries.** Public cross-module values use
   dataclasses from [src/schemas.py](src/schemas.py), not raw dicts, bare tuples,
   `TypedDict`, Pydantic models, or `**kwargs`.
8. **No logging framework.** Use `print()` for current status messages. Do not
   introduce `logging.basicConfig`, logger wrappers, or a custom logger class.
9. **Config-only experiments.** Switching model, category, corruption, threshold
   mode, or visualization mode must require only a config change.

---

## 3. Data and configuration rules

1. **Type precisely.** New dataclass fields and public function parameters must
   have concrete type annotations. Avoid `Any` except at unavoidable external
   boundaries, and validate immediately after parsing.
2. **Validate inputs.** User-controlled config, filesystem paths, dataset
   contents, model outputs, and external API/library results must be checked
   before use.
3. **Fail loudly on invalid config.** Unknown keys, wrong types, invalid enum
   values, empty paths, and non-finite numeric values should raise clear errors.
4. **Keep config schema synchronized.** Every new config field must be added to
   `config.yaml`, represented in `src/schemas.py`, validated when its domain is
   non-trivial, and documented in `README.md` when user-facing.
5. **No hardcoded secrets.** Never hardcode credentials, tokens, API keys, or
   private paths. This project should not need secrets for normal operation.
6. **No global mutable config.** Pass typed config objects through the existing
   flow instead of storing settings in module-level state.

---

## 4. Streaming and data flow

1. **Lazy iteration only.** `build_stream` and `apply_corruption` must remain
   generators. Do not materialize the full stream into a list.
2. **Determinism by seed.** Any code that rebuilds or re-iterates the stream
   must preserve the existing seed discipline.
3. **Preserve frame metadata.** Transformations that replace `Frame.image` must
   preserve `label`, `timestamp`, `source_id`, and `index`, preferably with
   `dataclasses.replace`.
4. **Keep image loading in `src/stream.py`.** Other modules should consume
   `Frame.image`, not perform independent dataset traversal or image I/O.
5. **Do not duplicate orchestration.** Per-frame dispatch to model, metrics, and
   visualization belongs in `main.py`.

---

## 5. Model rules

1. **Detector contract.** Every detector implements `fit_warmup(frames)` and
   `predict(frame) -> Prediction`.
2. **Inference is stable.** After warm-up, `predict` should not mutate model
   state unless the algorithm explicitly requires online adaptation and the
   behavior is documented.
3. **Finite outputs.** `Prediction.latency_ms` must be finite. Scores should be
   finite when possible; non-finite scores must be handled deliberately.
4. **Use existing libraries.** Reuse `anomalib` or another existing dependency
   for known SOTA detectors. Do not reimplement PatchCore, PaDiM, STFPM, and
   similar models from scratch without a specific reason.
5. **Device handling is config-driven.** Read `cfg.model.device`; do not
   hardcode `"cuda"` or `"cpu"`.
6. **New model changes stay local.** Add the class in `src/models.py`, register
   it in `build_model`, update config/docs as needed, and avoid new packages.

---

## 6. Corruption rules

1. **Severity is integer 1..3.** Do not introduce fractional or unbounded
   severity.
2. **Register kernels explicitly.** New kernels belong in `src/corruption.py`
   and `_CORRUPTIONS`.
3. **Kernel signature stays simple.** A corruption kernel takes
   `(img: np.ndarray, severity: int) -> np.ndarray` and returns a `uint8`
   `HxWx3` image.
4. **Severity tables are monotonic.** Three entries, ordered from mild to severe.
5. **Sampling is per spec per frame.** Multiple corruptions may compose. Do not
   silently change this to "pick one corruption".
6. **Preserve pass-through behavior.** When no corruption applies, yield the
   original `Frame` object.

---

## 7. Metrics and reporting rules

1. **Per-frame metric updates are O(1) amortized.** Do not store every score or
   latency just to compute final statistics.
2. **Snapshots work mid-stream.** `snapshot()` must remain callable before
   `finalize()`.
3. **Threshold logic is centralized in `main.py`.** Current modes are
   `max_score_ok` and `pot`. Adding a mode requires schema validation,
   calibration/reporting logic, and README documentation.
4. **Label handling is explicit.** Frames with unknown labels must not silently
   contaminate AUROC, AUPR, precision, recall, F1, or accuracy.
5. **Reports are self-describing.** `report.json` must include enough config,
   threshold, corruption, model, and runtime data to compare runs without
   external state.
6. **Report values are JSON-safe.** Convert numpy values, arrays, paths, and
   device objects before serialization.

---

## 8. Visualization rules

1. **Headless mode must stay headless.** `visualization.mode: file` and
   `visualization.mode: none` must not require a display server.
2. **Window-only UI dependencies.** Imports such as `cv2.imshow` behavior should
   remain isolated to window mode.
3. **Output layout is stable.** Do not move reports or frame outputs without
   updating README and architecture docs.
4. **No silent sampling.** `every_n_frames` is the visualization throttle. Do
   not add hidden frame skipping.

---

## 9. Workflow for any code change

1. **Check the worktree first.** There may be user changes. Do not revert or
   overwrite unrelated edits.
2. **Prefer editing existing files.** New source files require a clear reason
   and matching updates to [ARCHITECTURE.md](ARCHITECTURE.md) and
   [README.md](README.md).
3. **Keep docs truthful.** Update README/architecture only when behavior,
   public surfaces, config, outputs, or defaults changed.
4. **Run the smallest meaningful verification.** Prefer `python main.py` for
   behavior changes. Use targeted tests or import checks for narrow edits when a
   full run is too expensive.
5. **Verify determinism when randomness changes.** Same seed should produce the
   same semantic outputs, except expected runtime/latency fields.
6. **Verify corruption impact when corruption, model scoring, or metrics change.**
   Disabling vs enabling a severe corruption should affect quality metrics on a
   labelled dataset.
7. **Stop when done.** Once the requested behavior is implemented and verified,
   do not add polish, extra abstractions, unrelated cleanup, or speculative
   enhancements.

---

## 10. Anti-patterns to refuse

Reject or revise any diff that introduces:

- A second entry point.
- CLI flags for settings that belong in `config.yaml`.
- A `utils`, `helpers`, `common`, or `misc` module.
- A base class or plugin framework with one real implementation.
- Dead code, placeholder branches, or TODO scaffolding.
- Broad `try/except Exception` blocks that hide failures.
- Raw dicts, tuples, or `**kwargs` across module boundaries.
- Unvalidated external input, config, or model output.
- Hardcoded credentials, tokens, local absolute paths, or device names.
- Full-stream materialization for streaming data.
- Hidden state that makes repeated runs with the same seed diverge.
- New dependency managers or tooling config added incidentally.
- Documentation that describes behavior the code does not implement.
