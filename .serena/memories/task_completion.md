# Task Completion Checklist

## Before Committing Code

### 1. Code Quality
- [ ] Run black formatter: `black hummingbot/ --line-length 120`
- [ ] Run isort: `isort hummingbot/`
- [ ] Run flake8: `flake8 hummingbot/`

### 2. Testing
- [ ] Run tests: `make test`
- [ ] Ensure 80%+ coverage for new code
- [ ] Check diff coverage: `make development-diff-cover`

### 3. Build Verification
- [ ] Compile Cython: `./compile`
- [ ] Test in Docker: `docker compose up -d && docker attach hummingbot`

## For Script Strategies
- [ ] Test with paper_trade connector first
- [ ] Run `status` command to verify metrics display
- [ ] Check logs in `logs/` directory for errors

## Pull Request Checklist
- [ ] Branch from `development` (not main/master)
- [ ] Follow branch naming: `feat/`, `fix/`, `refactor/`, `doc/`
- [ ] Rebase with upstream: `git pull --rebase upstream development`
- [ ] Clear PR description with changes
- [ ] Enable "allow edits by maintainers"

## No Tests Required For
- UI components
- Configuration-only changes
- Documentation updates
