# Publishing a research software release

A release freezes a version of the MAVER framework that readers can inspect, install
and adapt. The manuscript's experimental results come from the research prototype;
identify that revision and its data archive separately from the framework release.

## Prepare the release commit

Use a pull request into `main`, including these metadata changes before merging:

- Keep `version` in `pyproject.toml` and `CITATION.cff` identical. Use the corresponding
  `vVERSION` tag for the release. If the package version changes, run `make lock`.
- Set `date-released` in `CITATION.cff` to the actual publication date (`YYYY-MM-DD`).
  Retain all four authors. Add a DOI only after one has been assigned to this artifact.
- Add a dated version summary to `CHANGELOG.md`, moving the released entries out of
  `Unreleased`. Preserve the detailed behavior-change history below it.

Validate the checkout:

```bash
uv sync --all-extras --locked
make check
make test-parity
uv run maver config check
uv run python examples/customisation.py
uv build
```

Parity checks report skips for documented divergences from frozen prototype inputs;
review those reasons when describing compatibility. PostgreSQL behavior is checked
by the required `postgres` CI job; local database validation uses `make test-postgres`
with `POSTGRES_TEST_DSN` pointing to a disposable database.

Confirm that both required CI jobs, `check` and `postgres`, pass for the pull
request's final revision, then merge it. Create the release from that commit on
`main`. Record its full SHA in the release notes, and retain the tag once cited.

The repository source archive is the primary artifact: it includes the configuration
files, benchmark adapters, examples, tests and dependency lock needed to adapt the
framework. A wheel is an optional installation artifact. Publishing to PyPI is not
required. If attaching build outputs, build them from a clean checkout of the release
commit and record their SHA-256 checksums.

## Suggested release notes

Adapt the following summary to the version being published, then add the final tag,
commit SHA and any artifact checksums:

> This release provides the MAVER framework for Kubernetes incident diagnosis and
> recovery. Typed diagnosis, planning and execution stages support optional
> verification and an experience-informed second diagnostic opinion. Python agent
> inheritance and workflows combine with YAML configuration; agents retain separate
> conversations within shared pipeline state, with scoped middleware prompt context
> and private metadata.
>
> Optional Kafka intake provides durable deduplication and quarantine. Restart recovery
> resumes eligible stages and parks uncertain execution for reconciliation. Human
> approvals persist across restarts. Classified Kubernetes API mutations are restricted
> to execution roles; `kubectl exec` requires human approval by default, including for
> diagnostic roles. Unattended SREGym execution requires an explicit opt-out.
>
> The `historical` text retrieval policy is selectable and enabled by the `ma-s` and
> `ma-s-v` arms. Execution accepts schema-supported string-encoded step arrays without
> consuming submission-guard retries. Benchmark configurations and model profiles are
> included. The manuscript's numerical results remain attributable to its research
> prototype and accompanying data archive.

## Citation and experimental provenance

Use the author list and title in the tagged `CITATION.cff`, together with the release
version, publication date and release URL. Include the full commit SHA when no stable
release identifier is available. Cite an assigned artifact DOI if available, and cite
the manuscript separately for its findings.

For a study using the framework, retain the source commit, dependency lock, selected
arm and model profile, resolved configuration with secrets removed, benchmark
revision and problems, model/serving revision and sampling parameters, and audit
records. Memory-enabled studies also need the corpus, record provenance and
leave-one-out exclusions. The [SREGym guide](adapters/sregym.md) describes historical
retrieval and unattended execution settings; [audit exports](audit.md) preserve run
evidence. These records identify the experimental condition even when stochastic
model outputs differ between runs.
