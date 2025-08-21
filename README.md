# gex

Common environment wrappers.

- `NormalizeEnv` – normalize observations and actions using provided statistics.
- `SuccessInfoWrapper` – rename `info['is_success']` to `info['success']`.
- `FrameStack` – stack the last `num_stack` observations along a new axis.
