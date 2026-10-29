

# Development

## Linting

```bash
$ nix develop -c ruff check --fix .
```

## Formatting

```bash
$ nix develop -c ruff format .
```

## Documentation

```bash
$ nix develop -c pydoctor --make-html src/
```

## Type Checking

```bash
$ nix develop -c ty check
```

## Testing

```bash
$ nix develop -c uv run pytest
```
