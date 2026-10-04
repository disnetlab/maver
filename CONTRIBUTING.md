# Contributing

Create a branch from `main`, make a focused change, and open a pull request into
`main`. Follow [STYLE_GUIDE.md](STYLE_GUIDE.md) for architectural conventions.
Report suspected vulnerabilities privately as described in
[SECURITY.md](SECURITY.md), rather than opening a public issue.

`main` requires a pull request, resolved review conversations, and successful
`check` and `postgres` CI jobs against the current base branch. These protections
also apply to administrators. Independent review approval is optional so a sole
maintainer can merge a tested pull request. Direct pushes, force pushes, and branch
deletion are blocked.

CI runs when a pull request targeting `main` is opened or updated. It can also be
started manually from GitHub Actions. Successful pull requests do not trigger a
second CI run after merging.

Before opening a pull request:

```bash
uv sync --all-extras --locked
make check
```

Database changes should also pass `make test-postgres` with `POSTGRES_TEST_DSN`
pointing to a test database. When changing dependencies, refresh both `uv.lock` and
`requirements.txt` with `make lock`; CI checks the exported requirements against
the lock.
