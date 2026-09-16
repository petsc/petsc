"""pde-sim Purple-agent binding for petscagent-bench.

Exposes the pde-sim multi-agent pipeline behind the A2A "Purple agent" contract
that petscagent-bench's Green agent expects, without modifying any bench code.

The Green agent talks to a Purple agent purely over A2A/HTTP by URL: it sends a
plain-text problem description and expects back a single message containing

  * exactly one TextPart shaped as
        Code generation successful ...
        nsize: <int>
        cli_args: <string>
  * one or more FileParts holding the generated PETSc source (first = main).

This server satisfies that contract by driving pde-sim non-interactively with
the Claude CLI (``claude -p`` in autonomous mode), then harvesting the generated
source and the run parameters from the pipeline's ``results-manifest.json``
(schema: .agents/pde-sim/contracts/results-manifest.schema.json).

Run it from an environment that has the A2A server deps installed (the simplest
is petscagent-bench's own uv env); see the sibling README.md for the full recipe.
"""

import argparse
import asyncio
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, FilePart, FileWithBytes, TextPart
from a2a.utils import new_agent_parts_message

# Repo root: this file lives at <repo>/.agents/pde-sim/bindings/petscagent-bench/pdesim_purple.py
REPO_DIR = Path(__file__).resolve().parents[4]

# Source suffixes forwarded to the bench's MCP uploader as compile dependencies
# (see petscagent-bench src/green_agent/agent.py::_create_files_on_server). A
# second ``.c`` would collide with the main after renaming, so only these extra
# kinds are forwarded as dependency files.
#
# KNOWN LIMITATION: the bench only *compiles* the dependency kinds it special-
# cases, and those are ``.cu`` and ``.kokkos.cpp`` — NOT ``.kokkos.cxx``. PETSc's
# own Kokkos convention is ``.kokkos.cxx`` (the tree has 28 of those and zero
# ``.kokkos.cpp``), so a generated ``.kokkos.cxx`` is uploaded but never compiled
# by the bench, yielding an undiagnosed link failure. Fixing that needs a change
# in the bench source, out of scope for this binding; until the bench recognizes
# ``.kokkos.cxx``, emit Kokkos dependencies as ``.kokkos.cpp``.
DEP_SUFFIXES = (".cu",)
DEP_COMPOUND_SUFFIXES = (".kokkos.cxx", ".kokkos.cpp")
MAIN_SUFFIXES = (".c", ".cxx", ".cpp", ".cu")

# Build-system drivers that must NOT appear as the executable in runs[].command.
# The bench compiles the source itself and runs the resulting binary directly, so
# the command must be the real invocation (e.g. 'mpiexec -n 4 ./app -opt val'),
# not a makefile target like 'make run' — parsing that would yield cli_args='run'
# and run the program default-configured, silently corrupting the bench score.
BUILD_TOOLS = {"make", "gmake", "bmake", "cmake", "ninja", "meson", "bear"}


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val else default


class Settings:
    """Runtime knobs, all overridable by environment variable."""

    def __init__(self) -> None:
        self.claude_bin = _env("PDESIM_CLAUDE_BIN", "claude")
        self.repo_dir = Path(_env("PDESIM_REPO_DIR", str(REPO_DIR)))
        self.artifacts_dir = Path(_env("PDESIM_ARTIFACTS_DIR", str(self.repo_dir / "artifacts")))
        # Keep under the bench's 50-minute Green->Purple read timeout (a2a_comm.py).
        self.timeout_sec = float(_env("PDESIM_TIMEOUT", "2700"))
        # Use os.environ.get, NOT _env: an explicit PDESIM_PERMISSION_FLAG="" must
        # DROP the flag (run under Claude's default, safer permissions), not fall
        # back to the default and silently reinstate bypassPermissions.
        self.permission_flag = os.environ.get("PDESIM_PERMISSION_FLAG", "--permission-mode bypassPermissions")
        self.claude_model = os.environ.get("PDESIM_CLAUDE_MODEL")  # optional --model
        self.extra_args = os.environ.get("PDESIM_CLAUDE_EXTRA_ARGS", "")  # optional passthrough
        # Bench rejects nsize outside [1, max_nsize] (default 64).
        self.max_nsize = int(_env("PDESIM_MAX_NSIZE", "64"))


SETTINGS = Settings()


def _build_prompt(problem_description: str, study_dir: Path) -> str:
    """Compose the autonomous pde-sim driver prompt."""
    return (
        "You are running the PETSc pde-sim multi-agent pipeline non-interactively as the "
        "code-generation target for an external benchmark.\n\n"
        "Phenomenon / problem description:\n"
        "<<<\n"
        f"{problem_description}\n"
        ">>>\n\n"
        "Instructions:\n"
        "- Follow the pde-sim pipeline defined by .claude/commands/pde-sim.md and "
        ".agents/pde-sim/skills/orchestration/SKILL.md.\n"
        "- Run with autonomy = autonomous: do NOT stop at any human-in-the-loop gate; make "
        "reasonable default choices and proceed to a verified PETSc program.\n"
        f"- Write ALL artifacts (problem-spec, numerical-plan, generated source, makefile, "
        f"results-manifest.json, logs) under this directory: {study_dir}\n"
        "- The deliverable is a compiled, MMS-verified PETSc program plus a results-manifest.json "
        "that conforms to .agents/pde-sim/contracts/results-manifest.schema.json, including at "
        'least one entry in "runs" with its full "command" and "mpi_ranks".\n'
        '- The "command" MUST be the direct program invocation (e.g. "mpiexec -n 4 ./app -opt val"), '
        'NOT a makefile target like "make run": the benchmark parses it to recover the executable and '
        "its CLI arguments, and a build-system target would be mis-run with the wrong arguments. Build "
        "through the makefile, but record the underlying run command here.\n"
        "- Prefer a single self-contained PETSc C source file for the main program.\n"
        "- Output discipline: an external grader compares the program's TRAILING numeric stdout "
        "lines against a hidden reference. Emit on stdout ONLY the program output the problem "
        "description explicitly asks for (e.g. the final solution via VecView()). Do NOT print extra "
        "numeric lines to stdout (MMS/convergence errors, norms, timings, iteration counts, "
        "residuals); send any such diagnostics to stderr or a file, or omit them. Match the "
        "description's stated grid/size, scheme, and boundary/initial conditions exactly.\n"
        "- Do NOT launch the visualization stage (it is opt-in per D20); no plots or renders are needed.\n"
        "- When finished, print EXACTLY one final line:\n"
        "      PDESIM_DONE <absolute path to results-manifest.json>\n"
    )


async def _run_claude(prompt: str, study_dir: Path) -> Tuple[int, str, str]:
    """Drive pde-sim via the Claude CLI; return (returncode, stdout, stderr)."""
    argv: List[str] = [SETTINGS.claude_bin, "-p", prompt]
    argv += shlex.split(SETTINGS.permission_flag)
    if SETTINGS.claude_model:
        argv += ["--model", SETTINGS.claude_model]
    if SETTINGS.extra_args:
        argv += shlex.split(SETTINGS.extra_args)

    print(f"@@@ pde-sim purple: launching pipeline (cwd={SETTINGS.repo_dir}, study={study_dir})", flush=True)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(SETTINGS.repo_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=SETTINGS.timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"pde-sim pipeline exceeded PDESIM_TIMEOUT={SETTINGS.timeout_sec:.0f}s")
    return proc.returncode, out_b.decode("utf-8", "replace"), err_b.decode("utf-8", "replace")


def _find_manifest(stdout: str, study_dir: Path, since_ts: float) -> Optional[Path]:
    """Locate the results-manifest.json produced by this run.

    Prefers the explicit ``PDESIM_DONE <path>`` marker the prompt asks for, then
    falls back to the newest results-manifest.json (mtime >= run start) under the
    requested study dir or the repo artifacts tree, since the orchestrator may
    place artifacts under artifacts/<study-id>/ instead.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("PDESIM_DONE"):
            cand = Path(line.split(None, 1)[1].strip()) if len(line.split(None, 1)) > 1 else None
            if cand and cand.is_file():
                return cand
            break

    search_roots = [study_dir, SETTINGS.artifacts_dir]
    newest: Optional[Path] = None
    newest_mtime = since_ts - 1.0
    for root in search_roots:
        if not root.exists():
            continue
        for m in root.rglob("results-manifest.json"):
            try:
                mt = m.stat().st_mtime
            except OSError:
                continue
            if mt >= since_ts and mt > newest_mtime:
                newest, newest_mtime = m, mt
    return newest


def _pick_run(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Choose a representative run: first successful, else first listed."""
    runs = manifest.get("runs") or []
    for r in runs:
        if r.get("status") == "success" and r.get("command"):
            return r
    for r in runs:
        if r.get("command"):
            return r
    return None


def _split_command(command: str) -> Tuple[Optional[str], str]:
    """From a run command, return (exe_basename, cli_args_string).

    e.g. 'mpiexec -n 4 ./poisson -pc_type gamg' -> ('poisson', '-pc_type gamg').

    The command must be the direct program invocation the bench will run, not a
    build-system target: a 'make run' command is rejected with ValueError so the
    caller reports the failure instead of mis-parsing 'run' as CLI arguments.
    """
    toks = shlex.split(command)
    exe_idx = next((i for i, t in enumerate(toks) if t.startswith("./")), None)
    if exe_idx is None:
        # No './exe' token: skip the launcher and any options it consumes, so the
        # value of a flag like '-n 4' is never mistaken for the executable.
        launchers = {"mpiexec", "mpirun", "srun", "petscmpiexec"}
        # Launcher flags that take a separate following value token.
        value_flags = {
            "-n", "-np", "--n", "--np", "-c", "-N", "--ntasks", "-ppn",
            "-host", "--host", "-hosts", "-hostfile", "--hostfile",
            "-machinefile", "-f", "-rf", "-wdir", "--wdir", "-path",
            "-bind-to", "-map-by", "-rmk", "-launcher", "-x", "-genv",
        }
        i = 1 if toks and os.path.basename(toks[0]) in launchers else 0
        while i < len(toks):
            t = toks[i]
            if t.startswith("-"):
                i += 2 if t in value_flags else 1
            else:
                break
        exe_idx = i if i < len(toks) else None
    if exe_idx is None:
        return None, ""
    exe = os.path.basename(toks[exe_idx])
    if exe.startswith("./"):
        exe = exe[2:]
    if exe in BUILD_TOOLS:
        raise ValueError(
            f"runs[].command invokes a build system ({exe!r}); it must be the direct "
            f"program invocation the bench runs (e.g. 'mpiexec -n 4 ./app -opt val'), "
            f"not a makefile target. Command was: {command!r}"
        )
    cli_args = " ".join(toks[exe_idx + 1:])
    return exe, cli_args


def _collect_sources(manifest_dir: Path, exe: Optional[str]) -> List[Tuple[str, bytes]]:
    """Gather the main source (+ recognized dependency sources) from the study dir."""
    sources: List[Tuple[str, bytes]] = []

    main_path: Optional[Path] = None
    if exe:
        for suf in MAIN_SUFFIXES:
            cand = manifest_dir / f"{exe}{suf}"
            if cand.is_file():
                main_path = cand
                break
    if main_path is None:
        # Fall back across every recognized main-source suffix, not just .c, so a
        # C++/Kokkos study (e.g. main.cxx) whose exe name does not match the source
        # stem is still found.
        candidates = sorted(p for suf in MAIN_SUFFIXES for p in manifest_dir.glob(f"*{suf}"))
        if len(candidates) == 1:
            main_path = candidates[0]
        elif candidates:
            # Ambiguous: prefer one whose stem appears in the exe name if known.
            main_path = next((p for p in candidates if exe and exe in p.stem), candidates[0])
    if main_path is None:
        return sources

    sources.append((main_path.name, main_path.read_bytes()))

    for p in sorted(manifest_dir.iterdir()):
        if not p.is_file() or p == main_path:
            continue
        name = p.name
        is_dep = name.endswith(DEP_COMPOUND_SUFFIXES) or p.suffix in DEP_SUFFIXES
        if is_dep:
            sources.append((name, p.read_bytes()))
    return sources


class PdeSimPurpleExecutor(AgentExecutor):
    """A2A executor that maps a problem description to a pde-sim study."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        problem_description = context.get_user_input()
        ctx = (context.context_id or "study").replace("/", "_")
        study_dir = SETTINGS.artifacts_dir / f"bench-{ctx}-{int(time.time())}"
        study_dir.mkdir(parents=True, exist_ok=True)

        try:
            since = time.time()
            prompt = _build_prompt(problem_description, study_dir)
            rc, stdout, stderr = await _run_claude(prompt, study_dir)
            print(f"@@@ pde-sim purple: pipeline exit code {rc}", flush=True)

            manifest_path = _find_manifest(stdout, study_dir, since)
            if manifest_path is None:
                raise RuntimeError(
                    "No results-manifest.json was produced by the pde-sim pipeline "
                    f"(exit={rc}). stderr tail: {stderr[-500:]!r}"
                )
            manifest = json.loads(manifest_path.read_text())
            manifest_dir = manifest_path.parent

            run = _pick_run(manifest)
            if run is None:
                raise RuntimeError(f"results-manifest.json at {manifest_path} has no usable run entry")

            exe, cli_args = _split_command(run["command"])
            nsize = int(run.get("mpi_ranks") or 1)
            nsize = max(1, min(nsize, SETTINGS.max_nsize))

            sources = _collect_sources(manifest_dir, exe)
            if not sources:
                raise RuntimeError(f"No generated source files found next to {manifest_path}")

            print(f"@@@ pde-sim purple: nsize={nsize} cli_args={cli_args!r} "
                  f"files={[n for n, _ in sources]}", flush=True)

            parts_list = [TextPart(text=f"Code generation successful ✅\nnsize: {nsize}\ncli_args: {cli_args}\n")]
            for name, data in sources:
                parts_list.append(FilePart(file=FileWithBytes(name=name, bytes=data, mime_type="text/plain")))
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )
        except Exception as e:  # noqa: BLE001 - report every failure over A2A
            print(f"@@@ pde-sim purple: ❌ {type(e).__name__}: {e}", flush=True)
            parts_list = [TextPart(text=f"Code generation failed ❌\nerror: {e}\n")]
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="pdesim_petsc_code_generation",
        name="pde-sim PETSc Code Generation",
        description="Drives the pde-sim multi-agent pipeline to turn a physical-phenomenon "
                    "description into a verified PETSc program, returned as source plus run args.",
        tags=["purple agent", "pde-sim", "PETSc", "multi-agent", "HPC"],
        examples=["Solve the 2-D steady Darcy flow problem on the unit square."],
    )
    return AgentCard(
        name="pdesim_purple",
        description="pde-sim binding for petscagent-bench: an A2A Purple agent that generates "
                    "verified PETSc code via the pde-sim pipeline.",
        url=url,
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/octet-stream"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


def start(host: str = "localhost", port: int = 9002, card_url: Optional[str] = None) -> None:
    card = prepare_agent_card(card_url or f"http://{host}:{port}")
    handler = DefaultRequestHandler(
        agent_executor=PdeSimPurpleExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    print(f"@@@ pde-sim purple: serving on http://{host}:{port} "
          f"(repo={SETTINGS.repo_dir}, claude={SETTINGS.claude_bin}, timeout={SETTINGS.timeout_sec:.0f}s)", flush=True)
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pde-sim Purple-agent binding for petscagent-bench.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="External URL to advertise in the agent card")
    args = parser.parse_args()
    start(host=args.host, port=args.port, card_url=args.card_url)
