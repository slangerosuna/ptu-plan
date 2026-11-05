# Repository Guidelines

## Project Structure & Module Organization

`ptu-plan` is a Rust 2024 project intended to load and solve HDDL planning domains and problems. It is currently a scaffold:

- `src/main.rs` contains the executable entry point and prints a greeting.
- `Cargo.toml` declares dependencies and library, planner, and training targets. The referenced `src/lib.rs` and `src/train.rs` do not exist yet.
- `deep-research-report.md` describes the proposed architecture, evaluation protocol, and implementation roadmap; treat it as design guidance, not implemented behavior.
- `target/` holds generated Cargo artifacts and is ignored by Git. No test, benchmark, or asset directories exist yet.

Keep reusable planning logic in focused library modules as they are introduced, with command-line parsing and orchestration in binary entry points.

## Build, Test, and Development Commands

Use the standard Rust toolchain:

- `cargo check` — check target definitions and type-check code.
- `cargo build` — build the package for local development.
- `cargo run --bin ptu-plan` — run the planner executable.
- `cargo test` — run unit, integration, and documentation tests.
- `cargo fmt -- --check` — check formatting with rustfmt.
- `cargo clippy --all-targets -- -D warnings` — lint all targets.

These are intended workflow commands, not a verified working build. Resolve the manifest's target definitions and missing source files before expecting package-wide checks to pass. The SCIP dependency enables a bundled native build, which may increase setup time.

## Coding Style & Naming Conventions

Use rustfmt defaults and four-space indentation. Name modules and functions with `snake_case`, types with `PascalCase`, and constants with `SCREAMING_SNAKE_CASE`. Prefer explicit types and documented invariants at solver and symbolic-representation boundaries.

## Testing Guidelines

Use Rust's built-in test framework: place unit tests in `#[cfg(test)]` modules and integration tests in `tests/`. Name tests after the behavior they verify. Add small, deterministic planning fixtures when parser and solver functionality arrives. No coverage threshold is currently configured.

## Commit & Pull Request Guidelines

There is no commit history or established message convention yet. Use concise, imperative subjects, such as `Add HDDL task parser`. Pull requests should describe the change, relevant design decisions, and validation results. Link related issues when available and identify any checks blocked by the scaffold.
