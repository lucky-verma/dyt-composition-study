# Public Release Checklist

Before flipping this repository public:

- [ ] Add arXiv URL and BibTeX entry.
- [ ] Re-run `python scripts/validate_repo.py`.
- [ ] Re-run `python scripts/smoke_model.py`.
- [ ] Confirm no internal machine names, paths, private W&B entities, or operational incident notes are present.
- [ ] Confirm large checkpoints are linked through a dataset/model host rather than committed to Git.
- [ ] Confirm README quick-start commands match the current code.
- [ ] Tag the first public release.

