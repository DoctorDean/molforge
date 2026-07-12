# Releasing molforge

molforge uses **PyPI trusted publishing** via GitHub Actions. New releases
are cut by pushing a `v*.*.*` tag; the `.github/workflows/release.yml`
workflow then builds the distribution, validates it, and uploads to PyPI.

## Cutting a release

1. **Decide on a version.** molforge follows
   [SemVer](https://semver.org/spec/v2.0.0.html):
   - `MAJOR.MINOR.PATCH`
   - Bump `PATCH` for fixes that don't change the API.
   - Bump `MINOR` for new features that are backward-compatible.
   - Bump `MAJOR` for breaking changes (rare in pre-1.0; we may still
     break things in `0.x` releases, but call it out loudly in the
     changelog).

2. **Update `src/molforge/__init__.py`**:
   ```python
   __version__ = "0.0.4"   # bump to whatever you're releasing
   ```
   The release workflow refuses to publish if this doesn't match the
   tag exactly.

3. **Update `CHANGELOG.md`**:
   - Move everything from `[Unreleased]` into a new section titled
     `[0.0.4] - YYYY-MM-DD`.
   - Add a fresh empty `[Unreleased]` header at the top.

4. **Commit**:
   ```bash
   git commit -am "release: 0.0.4"
   ```

5. **Tag and push**:
   ```bash
   git tag -a v0.0.4 -m "Release 0.0.4"
   git push origin master --tags
   ```

The release workflow runs on the tag push (not on the commit push to
master) and handles the rest:
- Builds sdist + wheel via `python -m build`.
- Verifies the wheel's `__version__` matches the tag.
- Runs `twine check` on the distributions.
- Installs the built wheel in a clean venv and imports `molforge`.
- Uploads to PyPI via trusted publishing.

## Dry-running the release workflow

To test the release pipeline without uploading to PyPI:

1. Go to **Actions -> Release -> Run workflow**.
2. Leave **Dry run** set to `true`.
3. Click Run workflow.

The workflow will build, validate, and smoke-test, but skip the upload
step.

## First-time setup (already done for molforge)

These steps were one-offs when the project was registered with PyPI;
documented here for posterity.

1. Create the package on PyPI by doing a single **manual** `twine upload`:
   ```bash
   python -m build
   twine upload dist/*
   ```
   Use an API token: username `__token__`, password `pypi-...`.

2. Configure trusted publishing on PyPI:
   - https://pypi.org/manage/account/publishing/
   - Add a new pending publisher for `molforge`:
     - PyPI project name: `molforge`
     - Owner: `DoctorDean`
     - Repository name: `molforge`
     - Workflow name: `release.yml`
     - Environment name: `pypi`

3. Create the `pypi` environment in GitHub:
   - **Settings -> Environments -> New environment**, name it `pypi`.
   - Optional: require approval before deploys.

After this, every tagged release publishes automatically with no
credentials in the repo.

## Troubleshooting

### "File already exists" on upload
PyPI does not allow re-uploading the same version. You'll need to
bump to a new version. Yanking the bad release is possible but
doesn't free the version number for re-use.

### "__version__ does not match tag"
The release workflow's version check failed. Update
`src/molforge/__init__.py` to match your tag, force-delete the tag
locally and on the remote, and re-tag.

```bash
git tag -d v0.0.4
git push origin :refs/tags/v0.0.4
# ... fix __version__, commit ...
git tag -a v0.0.4 -m "Release 0.0.4"
git push origin v0.0.4
```

### Build fails on missing files
Check that `pyproject.toml`'s `[tool.hatch.build]` section (or whatever
build backend is configured) includes all the data files you need.
`twine check` catches some packaging issues; the smoke-install step
catches more.
