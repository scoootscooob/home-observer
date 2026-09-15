# Security policy

This project handles camera and audio material on a local sensing host. Report
anything that could move media, paths, raw model output or credentials across the
production export boundary, or that could let the sandboxed frontier planner read
files or reach hosts it should not, by opening a private security advisory on the
GitHub repository. Please do not open a public issue for such reports.

The reference threat model and the enforced boundary are described in
`docs/production-privacy.md` and `docs/production-runtime.md`.
