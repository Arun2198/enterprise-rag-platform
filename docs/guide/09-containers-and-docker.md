# Chapter 9: Containers & Docker

This chapter starts the infrastructure half of the guide. If you already know what a container
is, skip to section 3 for this project's actual `Dockerfile`.

## 1. The problem containers solve

"It works on my machine" is a real, common failure mode: your laptop has Python 3.12, a specific
set of installed libraries, certain environment variables set, maybe a particular OS package
installed. The server you deploy to has none of that guaranteed to match. Traditionally, getting
an application to run reliably in a new environment meant manually replicating all of that setup
— easy to get subtly wrong, and it drifts over time as either environment changes.

A **container** solves this by packaging an application together with *everything it needs to
run* — the exact Python version, every installed library, any OS-level dependency — into one
self-contained unit that runs identically wherever it's executed: your laptop, a teammate's
laptop, a cloud server. It's not a full virtual machine (it doesn't emulate hardware or run a full
separate OS kernel — it shares the host machine's kernel but gets an isolated filesystem, process
space, and network namespace), which is why containers start in roughly a second and use a
fraction of the resources a VM would.

**Docker** is the dominant tool for building and running containers. Two core concepts:

- An **image** is the packaged, immutable snapshot — the filesystem, the installed dependencies,
  the startup command — built once from a recipe.
- A **container** is a running instance of an image — you can start several containers from the
  same image, each an independent, isolated running process.

The recipe for building an image is a **`Dockerfile`** — a plain text file of instructions,
executed top to bottom.

## 2. Layers, and why instruction order matters

Docker builds an image as a stack of **layers** — each instruction in the `Dockerfile` produces
one layer, and Docker caches each layer. If you rebuild an image and an early instruction hasn't
changed (same base image, same dependency list), Docker reuses the cached layer instead of
redoing that work — only instructions *after* the first actual change get re-executed. This is why
`Dockerfile`s conventionally copy dependency manifests and install dependencies *before* copying
application source code: source code changes far more often than dependencies do, so putting the
slow dependency-install step first means most rebuilds skip straight past it and only re-run the
fast "copy my code in" step.

## 3. This project's `Dockerfile`, line by line

```dockerfile
FROM python:3.12-slim
```
Start from an official, minimal Python 3.12 base image ("slim" — a smaller variant with fewer
preinstalled OS packages than the default Python image, keeping the final image size down).

```dockerfile
WORKDIR /app
```
All subsequent instructions run relative to `/app` inside the image's filesystem — creates the
directory if it doesn't exist and makes it the working directory.

```dockerfile
RUN pip install --no-cache-dir uv
```
Installs `uv` (the dependency manager this project uses — [Chapter 1](01-project-overview.md))
inside the image itself, so the rest of the build (and the container's runtime) can use it.
`--no-cache-dir` skips pip's own package cache, since it would just be dead weight in the final
image — nothing inside the container will `pip install` again later.

```dockerfile
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY sample_documents/ sample_documents/
```
Copies in the dependency manifest (`pyproject.toml` + the exact locked versions in `uv.lock`), the
actual application source, and the sample documents the demo/tests reference. `sample_documents/`
being included here specifically fixed a real deployment bug: an earlier version of this
`Dockerfile` omitted it, and the container image built from that version simply didn't have the
demo PDF available inside it at all.

```dockerfile
RUN uv sync --frozen
```
Installs every dependency exactly as pinned in `uv.lock` — `--frozen` means "fail rather than
silently re-resolve versions if the lockfile and `pyproject.toml` have drifted apart," which is
exactly the guarantee you want in a build: the container gets precisely what was tested, not
whatever the latest compatible versions happen to be on build day.

```dockerfile
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
```
Without this, the container's process runs as `root` by default — if the application had a
vulnerability that let an attacker execute arbitrary commands, they'd have root privileges *inside
the container*. Creating an unprivileged `appuser`, handing it ownership of `/app`, and switching
to it with `USER appuser` means the running application process has only the permissions it
actually needs. (Note: this reduces impact *inside* the container; it doesn't by itself prevent
container-escape vulnerabilities, which are a separate, deeper class of risk.)

```dockerfile
ENV PATH="/app/.venv/bin:$PATH"
```
`uv sync` creates a virtual environment at `/app/.venv`. Prepending its `bin/` directory to `PATH`
means commands like `uvicorn` (installed into that virtual environment) can be run directly by
name in the next instruction, without needing to activate the virtual environment explicitly.

```dockerfile
EXPOSE 8000
```
Documentation, not enforcement — declares that the containerized process listens on port 8000.
Doesn't itself publish the port to the host; that's a flag (`-p 8000:8000`) passed at
`docker run` time, or configured in the cloud deployment target ([Chapter 10](
10-aws-deployment.md)).

```dockerfile
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
```
The command that actually runs when a container starts from this image — launches the FastAPI app
via `uvicorn` (an ASGI server — the thing that actually accepts HTTP connections and calls into
the FastAPI application code), bound to `0.0.0.0` (all network interfaces inside the container,
not just `localhost` — required for anything outside the container to reach it) on port 8000.

## 4. What's deliberately excluded: `.dockerignore`

```
.git
.github
.venv
__pycache__
*.pyc
.pytest_cache
.vscode
.claude
tests/
evaluation/reports/
mlops_backups/
```

Same idea as `.gitignore`, but for what `COPY` instructions are allowed to pick up. This keeps the
built image smaller and avoids ever accidentally baking in things like local `.venv` state, git
history, or test artifacts that have no reason to exist inside a running container.

## 5. Building and running it yourself

```bash
docker build -t enterprise-rag-platform .
docker run -p 8000:8000 enterprise-rag-platform
```

The first command builds an image from the `Dockerfile` in the current directory, tagged
`enterprise-rag-platform`. The second starts a container from that image, mapping port 8000 on
your machine to port 8000 inside the container (matching the `EXPOSE`/`uvicorn --port` value
above) — after which `http://localhost:8000/health` should respond exactly as it would running
`uvicorn` directly, because it's the same application, just running inside an isolated,
reproducible environment instead of directly on your machine.

Next: [Chapter 10 — AWS Cloud Deployment](10-aws-deployment.md).
