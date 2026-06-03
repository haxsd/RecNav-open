# Artifact Policy

This public repository is code-first. It excludes files that are not suitable
for normal Git hosting:

- Habitat / HM3D / ObjectNav datasets
- model checkpoints and tokenizer weights
- generated videos, logs, and telemetry
- paper drafts and local planning notes
- third-party source trees cloned for local installation

Every experiment output should record:

- config or command line,
- seed,
- dataset split,
- checkpoint path,
- output path.

When reporting results, separate measured facts, interpretations, and known
limitations. Do not fill missing results by assumption.
