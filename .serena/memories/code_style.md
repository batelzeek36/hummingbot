# Hummingbot Code Style and Conventions

## Formatting

### Black (Python Formatter)
- Line length: **120 characters**
- Excludes: `.git`, `.venv`, `build`, `dist`, etc.

### isort (Import Sorting)
- Line length: 120
- Multi-line output style: 3 (vertical hanging indent)
- Include trailing comma: true
- Combine as imports: true

### flake8 (Linting)
- Max line length: 120
- Ignored rules: E251, E501, E702, W504, W503
- Special handling for `.pyx` and `.pxd` files

## Naming Conventions

### Files
- Snake_case for Python files: `my_strategy.py`
- Strategy scripts go in `scripts/` directory

### Classes
- PascalCase: `ScriptStrategyBase`, `OrderCandidate`
- Config classes end with `Config`: `SimplePMMConfig`

### Methods/Functions
- snake_case: `get_price_by_type()`, `create_proposal()`
- Async methods: Same convention, use `async def`

### Variables
- snake_case: `order_amount`, `bid_spread`
- Constants: UPPER_SNAKE_CASE

## Code Patterns

### Strategy Structure (Script-based)
```python
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from pydantic import Field
from hummingbot.client.config.config_data_types import BaseClientModel

class MyConfig(BaseClientModel):
    script_file_name: str = Field(default="my_script.py")
    exchange: str = Field(default="binance_paper_trade")
    # ... other config fields

class MyStrategy(ScriptStrategyBase):
    @classmethod
    def init_markets(cls, config: MyConfig):
        cls.markets = {config.exchange: {config.trading_pair}}
    
    def __init__(self, connectors, config):
        super().__init__(connectors)
        self.config = config
    
    def on_tick(self):
        # Main loop logic
        pass
```

### Type Hints
- Use type hints for function parameters and returns
- Use `Decimal` for prices and amounts (not float)
- Use `Dict`, `List`, `Optional` from typing module

### Imports
- Standard library first
- Third-party packages second
- Hummingbot imports last
- Use absolute imports within hummingbot

## Documentation
- Docstrings for classes and public methods
- Comments for complex logic
- Keep README files updated

## Commit Messages
- Present tense: "(feat) add unit tests"
- Prefixes: `(feat)`, `(fix)`, `(refactor)`, `(cleanup)`, `(doc)`
- ~70 characters for first line

## Branch Naming
- `feat/...` - new features
- `fix/...` - bug fixes
- `refactor/...` - code refactoring
- `doc/...` - documentation
