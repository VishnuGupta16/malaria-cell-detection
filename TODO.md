# Production Readiness — TODO

A checklist to take this data pipeline from working prototype to production-ready.
Grouped by concern; roughly ordered by effort-vs-payoff.

## Reproducibility
- [x] Pin exact dependency versions in `requirements.txt`
- [ ] Set a global random seed (TensorFlow + NumPy) so shuffle/augment are repeatable
- [x] Generate a full lockfile (`pip freeze`) for sub-dependencies — in `requirements.txt`
- [ ] Record the dataset version used (TFDS `malaria/1.0.0`)

## Configuration
- [x] Externalize settings in `config.py` (BATCH_SIZE, split ratios, DATA_DIR, VISUALIZE)
- [ ] Allow overrides via environment variables
- [ ] Keep secrets/credentials out of code (use env vars or a secrets manager)

## Observability
- [ ] Replace `print(...)` with the `logging` module (levels, timestamps)
- [ ] Log key pipeline stats (split sizes, class balance, timing per stage)
- [ ] Add metrics/monitoring hooks if run on a schedule or server

## Resilience
- [ ] Wrap dataset download/load in try/except with clear error messages
- [ ] Retry transient failures (e.g. download) with backoff
- [ ] Fail loudly on bad data instead of continuing silently

## Testing
- [ ] Unit test `split`: ratios are 70/15/15 and sets do NOT overlap
- [ ] Unit test `preprocess_image_dataset`: output shape 64x64x3, values in 0..1
- [ ] Unit test `augment_image`: output shape/labels preserved
- [ ] Add a smoke test that runs the pipeline on a tiny subset

## Data validation
- [ ] Verify image shapes and channels on load
- [ ] Detect/skip corrupt images
- [ ] Turn the class-balance check into a hard assertion (fail if skewed)

## Performance / scalability
- [ ] Add `.cache()`, `.prefetch()`, and `num_parallel_calls` to `.map()` steps
- [ ] Add real training batching (`.batch(n)`) separate from BATCH_SIZE (data-size cap)
- [ ] Confirm GPU usage when available

## Packaging & deployment
- [ ] Add `pyproject.toml` / proper package metadata
- [ ] Dockerfile for a reproducible runtime
- [ ] CI pipeline: run tests + lint on every push

## Documentation & maintainability
- [x] README with setup + run instructions
- [x] Single consolidated pipeline doc (`tasks/PREPROCESSING.md`)
- [ ] Docstrings + type hints on every function
- [ ] Lint/format config (ruff or flake8 + black)

## Security
- [ ] Scan dependencies for known vulnerabilities (e.g. `pip-audit`)
- [x] Ensure no secrets committed to git (`.gitignore` covers venv/, datasets/)
