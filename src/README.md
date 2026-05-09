# Source Layout

Reusable project code lives here. The repository-level scripts in `scripts/`
import these helpers directly, so the package does not need to be installed in
editable mode for normal use.

Current modules:

- `ua_detrac_starter.config` resolves config and dataset paths from repo root.
- `ua_detrac_starter.tlc_compat` patches 3LC/Ultralytics prediction format drift.
