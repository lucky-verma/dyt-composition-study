# Public Release Checklist

Before flipping or refreshing this repository for public release:

- [x] Add arXiv URL and BibTeX entry.
- [x] Re-run `python scripts/validate_repo.py`.
- [x] Re-run `python scripts/smoke_model.py` in an environment with PyTorch installed.
- [x] Confirm no internal machine names, paths, private W&B entities, or operational incident notes are present.
- [x] Confirm large checkpoints are linked through a dataset/model host rather than committed to Git.
- [x] Confirm README quick-start commands match the current code.
- [ ] Tag the first public release.
