"""Compiler front-ends for the Python port of win32fe (mirrors
compilerfe.cpp, clfe.cpp, iclfe.cpp).

Compiler.parse() removes arguments from the front of self.arg.  Handlers for
'-x' style options live in self.coptions and take any following arguments
with self.arg.popleft() (e.g. '-o <file>').
"""

import ntpath
import os
import sys
from collections import deque

import win32fe_utils as wfe
from win32fe_tool import Tool

COMPILER_HELP = """\
For compilers:
  win32fe will map the following <tool options> to their native options:
    -c:          Compile Only, generates an object file with .o extension
                 This will invoke the compiler once for each file listed.
    -l<library>: Link the file lib<library>.lib or if using --win-l also, <library>.lib
    -o <file>:   Output=<file> context dependent
    -D<macro>:   Define <macro>
    -I<path>:    Add <path> to the include path
    -L<path>:    Add <path> to the link path
    -g:          Generate debug symbols in objects when specified for compilation,
                 and in executables when specified for linking (some compilers
                 specification at both times for full debugging support).
    -O:          Enable compiletime and/or linktime optimizations.
Ex: win32fe cl -g -c foo.c --verbose -Iinclude

Note: win32fe passes the INCLUDE, LIB and PATH settings of its environment
      to the compiler.  Start Cygwin from a Visual Studio or oneAPI command
      prompt to set them.

=========================================================================
"""

CL_HELP = """\
cl specific help:
  win32fe uses -nologo by default for nonverbose output.  Use the flag:
     -logo  to disable this feature.
  -g is identical to -Z7.
  -O is identical to -O2.

=========================================================================
"""


class Compiler(Tool):
    """Base class for compiler front-ends (mirrors compilerfe.h)."""

    def __init__(self):
        super().__init__()
        self.compileoutflag = "-o"
        self.linkoutflag = "-o"
        self.output = None
        self.compilearg = deque()    # translated compile command args
        self.linkarg = deque()       # translated link-only args
        self.action = self.link      # compile or link (compiler::doit)
        self.coptions = {
            'D': self.found_D, 'I': self.found_I, 'L': self.found_L,
            'O': self.found_O, 'c': self.found_c, 'g': self.found_g,
            'l': self.found_l, 'o': self.found_o, 'm': self.found_m,
        }

    def parse(self):
        ierr = super().parse()
        if ierr:
            return ierr
        if self.arg:
            self.compilearg.appendleft(self.arg.popleft())  # the tool command
        while self.arg:
            temp = self.arg.popleft()
            if not temp.startswith("-"):
                self.found_file(temp)
            else:
                flag = temp[1] if len(temp) > 1 else ""
                h = self.coptions.get(flag)
                if h is None:
                    self.found_unknown(temp)
                else:
                    h(temp)
        return 0

    def help(self):
        super().help()
        print(COMPILER_HELP)
        return 0

    # -- compile / link (mirror compiler::Compile / compiler::Link) ----------
    def compile(self):
        """Invoke the compiler once per input file."""
        for i, f in enumerate(self.file):
            source = wfe.get_file_path(f, absolute=True)
            if source is None or wfe.get_short_path(source) is None:
                print("Error: win32fe: Input File Not Found: " + f)
                return -1
            output = self.output if i == 0 and self.output else ntpath.splitext(source)[0] + ".o"
            ierr = self.report([*self.compilearg, *self.output_args(self.compileoutflag, output), source])
            if ierr:
                return ierr
        return 0

    def link(self):
        files = list(self.file)
        objects = [i for i, name in enumerate(files) if name.endswith(".o")]
        if not objects:
            return self.report_link(files)

        import tempfile
        with tempfile.TemporaryDirectory(prefix="wfe") as directory:
            for i in objects:
                # Keep the base name because LINK reports it and names the default .exe after it
                os.mkdir(os.path.join(directory, str(i)))
                name = ntpath.splitext(ntpath.basename(files[i]))[0] + ".obj"
                dest = wfe.cygwin_conv_to_full_win32_path(os.path.join(directory, str(i), name))
                try:
                    if not wfe.copy_file(files[i], dest, self.verbose):
                        raise OSError("CopyFileW failed")
                except OSError as err:
                    print("Error: win32fe: cannot copy %s to %s: %s" % (files[i], dest, err), file=sys.stderr)
                    return 1
                files[i] = dest
            return self.report_link(files)

    def report_link(self, files):
        out = self.output_args(self.linkoutflag, self.output) if self.output else []
        argv = [*self.compilearg, *files, *out, *self.linkarg]
        if wfe.command_line_size(argv) <= wfe.MAX_COMMAND_LINE:
            return self.report(argv)
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix="wfe") as directory:
            inputs = wfe.response_file_args(directory, files)
            return self.report([*self.compilearg, *inputs, *out, *self.linkarg])

    @staticmethod
    def output_args(flag, path):
        return [flag, path] if flag == "-o" else [flag + path]

    # -- '-x' option handlers (mirror the compiler::Found* methods) ---------
    def found_D(self, temp):
        self.compilearg.append(temp)

    def found_I(self, temp):
        s = wfe.get_short_path(temp[2:])
        if s:
            self.compilearg.append("-I" + s)
        elif not self.woff:
            print("Warning: win32fe: Include Path Not Found: " + temp[2:])

    def found_L(self, temp):
        s = wfe.get_short_path(temp[2:])
        if s:
            self.linkarg.append("-L" + s)
        elif not self.woff:
            print("Warning: win32fe: Library Path Not Found: " + temp[2:])

    def found_O(self, temp):
        self.compilearg.append(temp)

    def found_c(self, temp):
        if temp == "-c":
            self.action = self.compile
        self.compilearg.append(temp)

    def found_g(self, temp):
        self.compilearg.append(temp)

    def found_l(self, temp):
        self.file.append(self.libprefix + temp[2:] + ".lib")

    def found_o(self, temp):
        if temp == "-o" and self.arg:
            out = self.arg.popleft()
            self.output = wfe.get_file_path(out, absolute=True)
            if self.output is None:
                print("Warning: win32fe: Output directory not found: " + ntpath.dirname(out))
                self.output = out
        else:
            self.compilearg.append(temp)

    def found_m(self, temp):
        print("Warning: win32fe: ignoring option: " + temp)

    def found_unknown(self, temp):
        self.compilearg.append(temp)

    def display_version(self):
        super().display_version()
        with wfe.OutputPipe(self.verbose) as out:
            if out.start([self.compilearg[0]]):
                out.get_exit_code()
                out.output_line()
                return 0
            return 1

    def is_a_known_tool(self):
        return True


class Cl(Compiler):
    """Microsoft C/C++ compiler front-end (mirrors clfe.cpp)."""

    def __init__(self):
        super().__init__()
        self.compileoutflag = "-Fo"
        self.linkoutflag = "-Fe"
        self.logo = False

    def parse(self):
        self.linkarg.appendleft("-link")
        ierr = super().parse()
        if not ierr and not self.verbose and not self.logo:
            self.compilearg.append("-nologo")
        return ierr

    def help(self):
        super().help()
        return self.cl_help()

    def cl_help(self):
        print(CL_HELP)
        return self.report([self.compilearg[0], "-?"])

    def report(self, argv):
        # hide "warning D4024: unrecognized source file type 'xx.o', ..."
        return self.report_filtered(
            argv,
            skip=lambda l: b"warning D4024" in l and b".o', object file assumed" in l)

    def found_L(self, temp):
        t = temp[2:]
        if t in ("D", "Dd"):
            self.compilearg.append(temp)
            return
        s = wfe.get_short_path(t)
        if s:
            self.linkarg.append("-libpath:" + s)
        elif not self.woff:
            print("Warning: win32fe: Library Path Not Found: " + t)

    def found_l(self, temp):
        if temp == "-logo":
            self.logo = True
        elif temp == "-link":
            while self.arg:             # the rest of the args go to the linker
                self.linkarg.append(self.arg.popleft())
        else:
            super().found_l(temp)

    def found_g(self, temp):
        self.compilearg.append("-Z7" if temp == "-g" else temp)

    def found_O(self, temp):
        self.compilearg.append("-O2" if temp == "-O" else temp)

    def found_unknown(self, temp):
        if temp == "-nologo":
            self.logo = False
        self.compilearg.append(temp)
