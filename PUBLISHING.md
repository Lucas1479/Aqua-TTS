# Publishing Aqua-TTS

This repository publishes `aqua-tts` through GitHub Actions and PyPI Trusted Publishing. Do not store PyPI API tokens in repository secrets.

## One-time setup

1. Create the GitHub environments `testpypi` and `pypi`.
2. Require manual approval for the `pypi` environment.
3. On TestPyPI, register a pending Trusted Publisher with:
   - PyPI project name: `aqua-tts`
   - Owner: `Lucas1479`
   - Repository: `Aqua-TTS`
   - Workflow: `publish.yml`
   - Environment: `testpypi`
4. Repeat on PyPI with environment `pypi`.

## Release checklist

1. Confirm the working tree contains only intentional release changes.
2. Update `aquatts.__version__` and `CHANGELOG.md`.
3. Build from a clean checkout (ignored `build/`, `dist/`, and `*.egg-info`
   directories must not be reused), then run:

   ```bash
   python -m ruff check aquatts tests
   python -m pytest tests -q
   python -m build
   python -m twine check dist/*
   ```

   Inspect the wheel and confirm it contains `aquatts/upstream.py` and the
   BigVGAN bridge, but no `_vendor/**/t2s_model.py`, `build_*`, `.pyd`, or
   compiler object files.

4. Merge the release commit and confirm CI is green.
5. Run the `Publish Python package` workflow with `testpypi`.
6. Install the TestPyPI artifact in a clean environment and verify `import aquatts`:

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "aqua-tts[runtime]"
   python -c "import aquatts; print(aquatts.__version__)"
   ```

7. Create and push the matching immutable Git tag, for example `v0.2.1`.
8. Run the workflow again from that tag with `pypi`, then approve the protected `pypi` environment.
9. Create the matching GitHub Release after the PyPI upload succeeds.

Package versions cannot be overwritten on PyPI. If an upload is wrong, increment the version before publishing again.
