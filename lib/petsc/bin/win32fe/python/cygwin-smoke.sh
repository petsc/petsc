#!/bin/sh
# cygwin-smoke.sh - smoke test for the Python win32fe on Cygwin/Windows.
#
# Run on the Cygwin machine, from anywhere:
#     sh lib/petsc/bin/win32fe/python/cygwin-smoke.sh
#
# Run the native checks for each available tool.
set -u
HERE=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
BIN=$(dirname -- "$HERE")
fails=0
ok()  { echo "PASS: $1"; }
bad() { echo "FAIL: $1"; fails=$((fails + 1)); }

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
if ! command -v "$PY" >/dev/null 2>&1; then echo "FAIL: no Python interpreter"; exit 1; fi

case "$(uname -s)" in
  CYGWIN*|MINGW*|MSYS*)
    ON_WINDOWS=1
    if command -v cygpath >/dev/null 2>&1; then ok "cygpath available"; else bad "no cygpath on Cygwin"; fi
    ;;
  *)
    ON_WINDOWS=0
    echo "SKIP: cygpath/compiler checks (uname = $(uname -s), not Cygwin/Windows)"
    ;;
esac

cd "$HERE" || exit
if "$PY" selftest.py; then ok "selftest.py"; else bad "selftest.py"; fi
if "$BIN/win32fe" --help >/dev/null; then ok "win32fe --help"; else bad "win32fe --help"; fi

test_compiler() (
  mkdir "$1" && cd "$1" && mkdir 'include dir' 'out dir' || exit 1
  printf '#define ANSWER 42\n' > 'include dir/answer.h'
  printf '#include <answer.h>\nint main(void){return NAME[1] == 32 ? ANSWER : 1;}\n' > 'my source.c'
  "$BIN/win32fe_$1" -c '-I"include dir"' '-DNAME="a b"' -o 'out dir/my file.o' 'my source.c' || exit 1
  printf 'keep this object\n' > sentinel
  cp sentinel 'out dir/my file.obj' || exit 1
  "$BIN/win32fe_$1" 'out dir/my file.o' -o 'my app.exe' && cmp sentinel 'out dir/my file.obj' || exit 1
  './my app.exe'
  [ "$?" -eq 42 ] || exit 1
  printf 'typedef int empty;\n' > empty.c
  "$BIN/win32fe_$1" -c empty.c -o empty.o || exit 1
  longtmp="$PWD/$(printf 'temp%0110d' 0)"
  mkdir "$longtmp" || exit 1
  TMPDIR="$longtmp" "$PY" -c 'import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:] + ["empty.o"] * 256))' \
    "$BIN/win32fe_$1" 'out dir/my file.o' -o long.exe || exit 1
  ./long.exe
  [ "$?" -eq 42 ] || exit 1
  printf 'const char *s = "caf\351";\n' > bytes.c
  "$1" -E -nologo bytes.c -link > direct.i 2>&1 &&
    "$BIN/win32fe_$1" -E bytes.c > wrapped.i 2>&1 && cmp direct.i wrapped.i
)

test_nvcc() (
  mkdir cuda && cd cuda || exit 1
  printf 'int main(void){return 42;}\n' > main.cu
  "$BIN/win32fe_nvcc" -g -c main.cu -o main.o || exit 1
  printf 'keep this object\n' > sentinel
  cp sentinel main.obj || exit 1
  "$BIN/win32fe_nvcc" main.o -L. -o app.exe && cmp sentinel main.obj || exit 1
  ./app.exe
  [ "$?" -eq 42 ]
)

test_archive() (
  mkdir archive && cd archive && mkdir expected || exit 1
  for i in 0 1 2 3 4 5 6 7 8; do
    printf 'int f%s(void){return %s;}\n' "$i" "$i" > "f$i.c"
  done
  cl -nologo -c f*.c || exit 1
  # The named LIB listing can represent only names in the Windows ANSI code page.
  if "$PY" -c 'import ctypes; "café cœur".encode("cp%d" % ctypes.CDLL("kernel32.dll").GetACP())' 2>/dev/null; then
    first=café.obj; last=cœur.obj
  else
    first=cafe.obj; last=coeur.obj
  fi
  mv f0.obj "$first" && "$BIN/win32fe_lib" -a 'my archive.lib' "$first" || exit 1
  "$BIN/win32fe_lib" -d 'my archive.lib' "$first" --version || exit 1
  set --
  for i in 1 2 3 4 5 6; do
    member=$(printf 'member %090d.obj' "$i")
    mv "f$i.obj" "$member" || exit 1
    set -- "$@" "$member"
  done
  "$BIN/win32fe_lib" -a 'my archive.lib' "$@" || exit 1
  # On code page 437, LIB prints cour.obj for both cour.obj and cœur.obj on stdout.
  mv f7.obj cour.obj && mv f8.obj "$last" || exit 1
  printf '%s\n' -nologo > flags.rsp
  "$PY" -c 'import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:-1] + ["@flags.rsp"] * 1200 + sys.argv[-1:]))' \
    "$BIN/win32fe_lib" -a 'my archive.lib' cour.obj "$last" || exit 1
  mv ./*.obj expected/ || exit 1
  "$BIN/win32fe_lib" --use lib.exe -x 'my archive.lib' || exit 1
  for member in expected/*.obj; do cmp "$member" "${member#expected/}" || exit 1; done
)

if [ "$ON_WINDOWS" -eq 1 ]; then
  tmp=$(mktemp -d) || exit 1
  trap 'rm -rf "$tmp"' EXIT
  cd "$tmp" || exit 1
  for tool in cl icx; do
    if ! command -v "$tool" >/dev/null 2>&1; then echo "SKIP: $tool";
    elif test_compiler "$tool"; then ok "$tool compile, link, run and preprocess";
    else bad "$tool compile, link, run or preprocess"; fi
  done
  if ! command -v nvcc >/dev/null 2>&1; then echo "SKIP: nvcc";
  elif test_nvcc; then ok "nvcc link preserves main.obj"; else bad "nvcc link"; fi
  if ! command -v cl >/dev/null 2>&1 || ! command -v lib.exe >/dev/null 2>&1; then echo "SKIP: LIB";
  elif test_archive; then ok "LIB update and extraction"; else bad "LIB update or extraction"; fi
fi

cd "$HERE" || exit
if [ "$fails" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "$fails CHECK(S) FAILED"; fi
exit "$fails"
