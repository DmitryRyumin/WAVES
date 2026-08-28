app_port := "7860"

css_source := "styles/app.css"
css_output := "app.css"


default:
    @just --list


sync:
    uv sync


sync-node:
    npm ci


setup: sync sync-node


lint:
    uv run ruff check .


format-python:
    uv run ruff format .


format-css:
    npm run lint:css:fix


format-web:
    npm exec -- prettier --write .


format-toml:
    npm exec -- taplo fmt


format-readme:
    uv run python scripts/update_readme.py


format: format-python format-css format-web format-toml format-readme


css-build:
    npm run build:css


css-build-check:
    @tmp="$(mktemp "${TMPDIR:-/tmp}/waves-css.XXXXXX")"; \
    trap 'rm -f "$tmp"' EXIT; \
    npm exec -- lightningcss \
        --bundle \
        --minify \
        --browserslist \
        "{{css_source}}" \
        -o "$tmp"; \
    if ! cmp -s "$tmp" "{{css_output}}"; then \
        echo "Generated {{css_output}} is out of date."; \
        echo "Run: just css-build"; \
        exit 1; \
    fi


typecheck:
    uv run mypy


deptry:
    uv run deptry .


check-python:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run deptry .


check-css:
    npm run lint:css


check-web:
    npm exec -- prettier --check .


check-toml:
    npm run lint:toml
    npm exec -- taplo fmt --check


check-readme:
    uv run python scripts/update_readme.py --check


check: check-python check-css check-web check-toml check-readme css-build-check


fix:
    uv run ruff check . --fix
    @just format
    @just css-build


stop:
    @pid="$(lsof -tiTCP:{{app_port}} -sTCP:LISTEN 2>/dev/null || true)"; \
    if [ -n "$pid" ]; then \
        echo "Stopping process on port {{app_port}}: $pid"; \
        kill $pid 2>/dev/null || true; \
        for _ in 1 2 3 4 5; do \
            if ! lsof -tiTCP:{{app_port}} -sTCP:LISTEN >/dev/null 2>&1; then \
                break; \
            fi; \
            sleep 0.2; \
        done; \
        pid="$(lsof -tiTCP:{{app_port}} -sTCP:LISTEN 2>/dev/null || true)"; \
        if [ -n "$pid" ]; then \
            echo "Force stopping process on port {{app_port}}: $pid"; \
            kill -9 $pid 2>/dev/null || true; \
        fi; \
    fi


run: css-build stop
    @if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then \
        DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
            uv run python app.py; \
    else \
        uv run python app.py; \
    fi


dev: css-build stop
    @if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then \
        DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
            uv run gradio app.py; \
    else \
        uv run gradio app.py; \
    fi