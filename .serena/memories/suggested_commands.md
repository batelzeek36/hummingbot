# Suggested Commands for Hummingbot Development

## Running Hummingbot

### Docker (Recommended)
```bash
# Start container
docker compose up -d

# Attach to running instance
docker attach hummingbot

# View logs
docker logs hummingbot

# Stop container
docker compose down
```

### Source Installation
```bash
# Install (creates conda env)
./install

# Compile Cython extensions
./compile

# Run Hummingbot
./start

# Clean build artifacts
./clean

# Uninstall
./uninstall
```

## Inside Hummingbot CLI
```
start --script <script_name>.py    # Start a script strategy
start --script <script>.py --config <config>.yml  # With config
status                              # View current status
stop                                # Stop strategy
history                            # View trade history
exit                               # Exit Hummingbot
```

## Development Commands

### Testing
```bash
# Run all tests with coverage
make test

# Run specific test file
python -m pytest test/hummingbot/path/to/test.py -v

# Run with coverage report
make run_coverage

# Check diff coverage against development
make development-diff-cover
```

### Code Quality
```bash
# Format with black (line-length 120)
black hummingbot/ --line-length 120

# Sort imports with isort
isort hummingbot/

# Lint with flake8
flake8 hummingbot/
```

### Building
```bash
# Compile Cython extensions
./compile
# or
make build

# Clean and rebuild
make clean && make build

# Build Docker image
make docker
```

## Git Workflow
```bash
# Add upstream remote
git remote add upstream https://github.com/hummingbot/hummingbot.git

# Create feature branch from development
git checkout -b feat/my-feature upstream/development

# Rebase with upstream
git pull --rebase upstream development
```

## Useful Darwin (macOS) Commands
```bash
# Check Docker status
docker ps -a

# Find files
find . -name "*.py" -type f

# Search in files
grep -r "pattern" hummingbot/

# List directory structure
ls -la
```
