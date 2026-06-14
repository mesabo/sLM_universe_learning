# Exercises — Class 1: Tools and Function Calling

## Warm-up: Inspect tool schemas

Call `add.args` (a dict of field name → type) and `add.get_input_schema().schema()` (the full JSON schema). Print them. Then manually construct a tool call: `add.invoke({"a": 100.0, "b": 200.0})` and verify the result is `300.0`.

## Apply: Tool with side effects

Create a tool that maintains state between calls:

```python
_counter = {"value": 0}

@tool
def increment(amount: int) -> int:
    """Increment the internal counter by amount and return the new value."""
    _counter["value"] += amount
    return _counter["value"]
```

Invoke `increment(5)` three times and verify the counter reaches `15`. Discuss why tools with side effects need careful design in multi-agent settings.

## Stretch: Dynamic tool registry

Implement a `ToolRegistry` class that stores tools by name and supports runtime registration:

```python
class ToolRegistry:
    def __init__(self): self._tools = {}
    def register(self, tool): self._tools[tool.name] = tool
    def get(self, name): return self._tools.get(name)
    def all(self): return list(self._tools.values())
```

Register `add`, `multiply`, and `sqrt`. Then implement a `dispatch(name, args)` method that looks up the tool by name and calls it. Use this registry instead of the hardcoded tools list in `bind_tools`. Discuss how this pattern supports plugin architectures for production agents.
