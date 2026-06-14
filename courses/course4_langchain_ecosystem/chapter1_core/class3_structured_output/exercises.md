# Exercises — Class 3: Structured Output

## Warm-up: Add a field

Extend the `Movie` schema with an `is_sequel: bool` field. Update the prompt to include "and indicate whether it is a sequel." Run the chain on a known sequel (e.g., "The Dark Knight (2008), a superhero film, is the sequel to Batman Begins.") and verify `is_sequel == True` in the parsed output.

## Apply: Nested schema

Define a nested Pydantic model:

```python
class Actor(BaseModel):
    name: str
    role: str

class FilmCast(BaseModel):
    film_title: str
    actors: list[Actor]
```

Build a chain that extracts cast information from a short text description. Verify that the parsed result has `film_title` and at least one actor in `actors`.

## Stretch: Retry with correction

Implement a retry loop using `with_retry(stop_after_attempt=3)` combined with a fix-up prompt. On a `json.JSONDecodeError`, inject the raw malformed output back into a second prompt: "Fix this malformed JSON to match the schema: {raw}". Measure how often the fix-up prompt succeeds vs the original prompt for a local sLM.
