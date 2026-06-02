# Environment Layout

JAMEL keeps environment assets under the top-level `env/` directory. This
directory is for runnable environment resources and downloaded static files, not
for Python packages.

Current layout:

```text
env/
  browser_env/
    scalewob-env/       # downloaded ScaleWoB static apps; not committed
```

Future environment families should be added as siblings:

```text
env/
  browser_env/
  mobile_env/
  game_env/
  embodied_env/
```

Python integration code remains under `jamel/core/env/`. Large environment
assets, generated caches, and downloaded files should stay under `env/<name>/`
and be excluded from source control when they are reproducible or too large.

Browser-specific setup is documented in [ENVIRONMENT.md](ENVIRONMENT.md).
