# win32fe Python port

A Python port of [PETSc's win32fe](https://bitbucket.org/petsc/win32fe) that runs under
**Cygwin Python**. No compilation step is needed. The `src/` and `include/`
paths below refer to the C++ sources in that repository.

PETSc contributions, including this Python port, use the repository’s [BSD 2-Clause license](../../../../../LICENSE).
The original C++ win32fe retains its [original copyright notice](https://bitbucket.org/petsc/win32fe/src/master/readme.html).

## Layout

`lib/petsc/bin/win32fe/` holds the front-end scripts: `win32fe`, which
takes the place of `win32fe.exe`, and the `win32fe_*` wrappers. The Python
modules, tests and this README live in its `python/` subdirectory:

| file in `python/` | mirrors |
|---|---|
| `win32fe_frontends.py` | `src/create.cpp`, `src/win32fe.cpp` (tool registry `TOOLS` and `main()`) |
| `win32fe_tool.py` | `src/toolfe.cpp`, `include/toolfe.h` (base `Tool` class) |
| `win32fe_compiler.py` | `src/compilerfe.cpp`, `src/clfe.cpp`, `src/iclfe.cpp` |
| `win32fe_archiver.py` | `src/archiverfe.cpp`, `src/libfe.cpp` |
| `win32fe_utils.py` | `src/utils.cpp`, `src/win32fepath.c`, `src/mycygpath.cpp`, `src/outpipe.cpp` |
| `selftest.py` | argument translation and process output tests (`python3 selftest.py`) |

Parsing convention: parse loops pop arguments off the front of `self.arg`;
option handlers consume any following arguments they need with
`self.arg.popleft()` (e.g., `-o <file>`).

## Usage (same options as the C++ version)

    win32fe <tool> --<win32fe options> -<tool options> <files>
    # Compile and link in one step:
    win32fe cl foo.c -Iinclude -Llibdir -lfoo -o app

    # Compile and link in separate steps:
    win32fe cl -c foo.c -Iinclude -o foo.o
    win32fe cl foo.o -Llibdir -lfoo -o app

`<tool>`: cl, icl, ifort, lib, nvcc. Intel `icx` and `ifx` run as
`icl --use icx` and `ifort --use ifx`.

## How the Windows-specific parts map to Python

- POSIX -> Win32 path conversion: under Cygwin, calls `cygwin_conv_path()`
  from `cygwin1.dll` through `ctypes`. That is the conversion `cygpath -aw`
  does, and `src/mycygpath.cpp` ran `cygpath -aw` as a separate process for
  each path. Elsewhere the port runs `cygpath -aw` if it is on `PATH`. Short
  (8.3) paths via `GetShortPathNameW` and long
  paths via `GetLongPathNameW` through `ctypes` on kernel32 (mirrors
  `src/win32fepath.c`). `GetShortPathNameW` failing doubles as the
  existence check, as in the C++ code.
- Spawning the real tool: the front ends keep filenames and options in an
  argument list. `OutputPipe` quotes it once for `CreateProcessW` with
  `list2cmdline()` in `win32fe_utils.py`, which follows the rules of
  `subprocess.list2cmdline()` without importing `subprocess` on every
  compiler call. It captures `stdout` and `stderr` in a temporary file, and
  obtains the exit code with `WaitForSingleObject` and `GetExitCodeProcess`.
- Environment of the real tool: a Cygwin process has only `PATH`, `SYSTEMDRIVE`,
  `SYSTEMROOT` and `WINDIR` in its Windows environment, which `CreateProcessW`
  passes to the tool. Under Cygwin the port first copies the Cygwin
  environment (`INCLUDE`, `LIB`, `TMP`, ...) into it with
  `cygwin_internal(CW_SYNC_WINENV)`. The C++ `win32fe.exe` is a native
  program, so Cygwin gave it the full environment.
- File ops, temp files, env vars: kernel32 via `ctypes` (`CreateFileW`,
  `CopyFileW`, `DeleteFileW`, `GetTempPathW`, `GetTempFileNameW`,
  `GetEnvironmentVariableW`, `SetEnvironmentVariableW`, ...). `--path`
  updates the Windows `PATH` with `SetEnvironmentVariableW`, as the C++ does,
  and leaves the POSIX `PATH` in `os.environ` unchanged.

## Changes from the C++ win32fe

The port accepts the win32fe 1.11.4 wrapper options and returns the tool's
exit code. The changes in behavior are below.

### Packaging and entry points

- Python modules replace the prebuilt `win32fe.exe` and `win32feutils.dll`.
  No compilation step is needed.
- `win32fe` is a Python script at the old `win32fe.exe` location, so
  `${PETSC_DIR}/lib/petsc/bin/win32fe/win32fe --help` still works.
- The `win32fe_*` wrappers are Python scripts that call `main()` in the same
  process. The old sh wrappers started a shell, which then started
  `win32fe.exe`.
- The wrappers pass each argument unchanged. The old wrappers expanded an
  unquoted `$*`, which split arguments that contain spaces.
- `win32fe_lib` replaces each `qc` argument from Cygwin CMake with `-a`. The
  old wrapper ran `sed` on the joined argument string.
- The scripts run `/usr/bin/python3 -S`, so they need a 64-bit Cygwin Python,
  version 3.6 or later. The port loads kernel32 with `ctypes.CDLL`, which is
  safe only with the single calling convention of 64-bit Windows.
- `win32fe --version` reports `Win32 Development Tool Front End (Python port),
  version 1.11.4-py` and a date. It keeps the `Win32 Development Tool Front
  End` text that PETSc's configure looks for.

### Supported tools

- `cl`, `icl`, `ifort`, `lib` and `nvcc`. Intel `icx` and `ifx` run as
  `icl --use icx` and `ifort --use ifx`, as before.

### Behavior

- Under Cygwin, paths convert in-process with `cygwin_conv_path()` from
  `cygwin1.dll`. The C++ started a `cygpath -aw` process for each path. A
  compile now starts only the compiler.
- When linking, the port copies `foo.o` to `foo.obj` in a private temporary
  directory. The C++ `cl`, `icl` and `ifort` front ends used a unique name
  next to `foo.o`. The C++ `nvcc` front end wrote `foo.obj` next to `foo.o`,
  overwriting any existing `foo.obj`, and deleted the copy after linking.
  If the translated link command exceeds the Windows limit, win32fe passes
  the input filenames to the driver in response files.
- `nvcc` receives its own `-g` and `-L` options instead of `-Z7` and `-libpath:`.
  `win32fe nvcc --help` displays the CUDA driver's help.
- `lib -a` uses temporary response files for long member lists. Caller-supplied
  `@file` arguments retain their position and stay outside generated response files.
  If the response-file paths exceed the Windows command-line limit, win32fe updates the archive in batches.
- `lib -x` without member names asks the selected librarian to write a listing to a file.
  Microsoft LIB uses the Windows ANSI code page for that file, independent of the console code page.
  For names that this code page cannot represent, specify the original member names with `-x`.
- `--help` and `--version` run only the requested information command, even when the arguments include a compile or archive operation.
- The wrapper forwards unfiltered tool output byte for byte, including
  preprocessor output, line endings and an unterminated last line.
- If the tool cannot start, the port always prints
  `win32fe: failed to execute:` and the command to `stderr`. The C++ printed
  this only with `--verbose`.
- An unknown tool name prints an error to `stderr` and exits with status 1.
  The C++ printed the error to `stdout` and exited with status 0.
- `--wait_for_debugger` prints the wrapper's Windows PID and waits for Enter before it starts the native tool. The C++ looped
  until a debugger changed a variable.
- The compiler `--help` text says that win32fe passes `INCLUDE`, `LIB` and `PATH`
  from its environment to the compiler. The C++ text still said that win32fe
  finds the system include and library paths, which it stopped doing in 2013.
- A trailing `-o` without a file name passes through to the tool. The C++
  read past the end of the argument list. An `nvcc` link input without a file
  extension no longer aborts win32fe.

### Not ported

- The front ends for Compaq Visual Fortran (`df`, `f90`), Borland C++
  (`bcc32`, `tlib`) and the Intel Fortran from before `ifort` (`ifl`). PETSc
  no longer supports these compilers.
- The registry lookup (`GetStringValueFromRegistry`) and `SearchPath` from
  `utils.cpp`. Nothing in the C++ called them after it removed its compiler
  autodetection in 2013.
- The Windows NT4 short-path code (`win32fe_GetShortPath_nt4`). `--NT4` is
  still accepted and does nothing, as in the C++.

"How the Windows-specific parts map to Python" describes the mechanisms
behind these changes, including the Windows environment that Cygwin passes to
native tools.

## Testing

    python3 lib/petsc/bin/win32fe/python/selftest.py

records argument lists before launch to check flag translation, quoted paths,
temporary objects, response files, and archive member selection. It also
launches a Python child to verify argument boundaries and output bytes. No
compiler or librarian is required, and the test runs without Windows, for
example on Linux: when `kernel32` is unavailable, the Win32 calls fall back to
`os` and `subprocess`. Under Cygwin win32fe always uses the Win32 calls.

## Wrapper scripts (in-process)

PETSc's `lib/petsc/bin/win32fe/win32fe_cl` etc. are Python scripts like
`win32fe` itself. Each one calls `main()` with its tool name, without
spawning another process. They run `/usr/bin/python3 -S` directly: no
`env` process, and no `site` import, which the port does not need:

    #!/usr/bin/python3 -S
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "python"))
    from win32fe_frontends import main
    sys.exit(main([sys.argv[0], "cl"] + sys.argv[1:]))

`main([sys.argv[0], "icl", "--use", "icx"] + args)` is the equivalent of
`win32fe icl --use icx <args>`.

## Testing on Cygwin/Windows

    sh lib/petsc/bin/win32fe/python/cygwin-smoke.sh

runs the portable tests first. With `cl` or `icx` on `PATH` it compiles and
links files with spaces in their names, runs the executable, and compares
direct and wrapped preprocessor output. It also links object lists that exceed the Windows command-line limit.
With `nvcc` it checks that linking preserves an existing `.obj`. With `cl` and `lib.exe`
it updates an archive from a long list of response files and compares the extracted objects with the originals.
