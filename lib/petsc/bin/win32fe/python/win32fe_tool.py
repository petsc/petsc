"""Base tool front-end for the Python port of win32fe (mirrors toolfe.cpp /
toolfe.h).

Parsing convention: the parse loops remove arguments from the front of
self.arg.  An option handler takes any following arguments it needs with
self.arg.popleft().  Tool.parse() keeps the non-'--' arguments in order for
the subclass parse step (Compiler/Archiver).
"""

import os
import sys
from collections import deque

import win32fe_utils as wfe

VERSION = "1.11.4-py"
BUILD_DATE = "2026-11-18"

TOOL_HELP = """\
Usage: win32fe <tool> --<win32fe options> -<tool options> <files>
  <tool> must be the first argument to win32fe

<tool>: {cl,icl,ifort,lib}
  cl:    Microsoft 32-bit C/C++ Optimizing Compiler
  icl:   Intel C/C++ Optimizing Compiler (icl --use icx for icx)
  ifort: Intel Fortran Optimizing Compiler (ifort --use ifx for ifx)
  lib:   Microsoft Library Manager

<win32fe options>:
  --help:       Output this help message and help for <tool>
  --path <arg>: <arg> specifies an addition to the PATH that is required
                (ex. the location of a required .dll)
  --use <arg>:  <arg> specifies the variant of <tool> to use
  --verbose:    Echo to stdout the translated commandline
                and other diagnostic information
  --version:    Output version info for win32fe and <tool>
  --wait_for_debugger:  Print the Windows PID of win32fe and wait for Enter
                before starting the tool.
  --win-l:      For compilers, define -lfoo to link foo.lib instead of libfoo.lib
  --woff:       Suppress win32fe specific warning messages
=================================================================================
"""


class Tool:
    """Base class: parses the win32fe '--' options and dispatches the rest."""

    def __init__(self):
        self.options = {
            "--help": self.found_help,
            "--path": self.found_path,
            "--use": self.found_use,
            "--verbose": self.found_verbose,
            "--version": self.found_version,
            "--woff": self.found_woff,
            "--NT4": self.found_nt4,
            "--wait_for_debugger": self.found_wait,
            "--win-l": self.found_win_l,
        }
        self.arg = deque()          # remaining arguments
        self.file = deque()         # input files
        self.version_string = ("Win32 Development Tool Front End (Python port), "
                               "version " + VERSION + " " + BUILD_DATE)
        self.verbose = False
        self.woff = False
        self.libprefix = "lib"      # -lfoo -> libfoo.lib ("--win-l" -> foo.lib)
        self.info_action = None     # explicit --help or --version
        self.action = None          # operation selected by the front end
        self._use = None            # replacement tool path from --use

    # -- argument intake (mirrors tool::GetArgs) ----------------------------
    def get_args(self, argv):
        if len(argv) == 1:
            self.arg.append("--help")
            return
        for temp in argv[1:]:
            if temp == "--wait_for_debugger":
                # Wait for Enter where the C++ spins until a debugger flips a variable
                print("Waiting for debugger to attach.....")
                k = wfe.kernel32()
                print("PID = %d" % (k.GetCurrentProcessId() if k is not None else os.getpid()))
                print("Attach a debugger now, then press Enter to continue.")
                try:
                    input()
                except EOFError:
                    pass
            elif temp == "--verbose":
                self.verbose = True
                print()
            self.arg.append(temp)
        if len(argv) == 2 and self.arg[0] not in ("--version", "--help"):
            self.arg.append("--help")
        if len(argv) == 3 and self.arg[-1] == "--verbose":
            self.arg.append("--help")

    # -- parse / execute (mirrors tool::Parse / tool::Execute) --------------
    def parse(self):
        rest = deque()
        while self.arg:
            temp = self.arg.popleft()
            if temp.startswith("--"):
                h = self.options.get(temp)
                if h is None:
                    rest.append(temp)       # keep unknown -- options for the tool
                else:
                    h(temp)                 # may popleft() values from self.arg
            else:
                rest.append(temp)
        if self._use is not None:           # --use replaces the tool name
            if rest:
                rest[0] = self._use
            else:
                rest.append(self._use)
        self.arg = rest
        if not self.is_a_known_tool() and self.info_action is None:
            return self.unknown_tool(self.arg[0] if self.arg else "")
        return 0

    @staticmethod
    def unknown_tool(name):
        print("Error: win32fe: Unknown Tool: " + name, file=sys.stderr)
        print("  Use --help for more information on win32fe options.", file=sys.stderr)
        return 1

    def execute(self):
        if self.info_action is not None:
            return self.info_action()
        if self.verbose:
            ierr = self.display_version()
            if ierr:
                return ierr
        return self.action() if self.action is not None else 0

    # -- running the real tool ------------------------------------------------
    def report(self, argv):
        return self.report_filtered(argv)

    def report_filtered(self, argv, skip=None):
        """Filter known diagnostics unless verbose output includes warnings."""
        with wfe.OutputPipe(self.verbose) as out:
            ierr = out.execute(argv)
            out.output_all(None if self.verbose and not self.woff else skip)
            return ierr

    # -- help / version --------------------------------------------------------
    def help(self):
        print()
        print(self.version_string)
        print()
        print(TOOL_HELP)
        return 0

    def display_version(self):
        print(self.version_string)
        return 0

    # -- '--' option handlers (mirror the tool::Found* methods) ---------------
    def found_path(self, temp):
        # Usage: --path "path1;path2;path3"
        if not self.arg:
            self.arg.append("--help")
            return
        inpath = self.arg.popleft()
        shortpath = ""
        for pathi in inpath.split(";"):
            s = wfe.get_short_path(pathi)
            if s:
                if self.verbose:
                    print("win32fe: Adding to path: " + wfe.get_long_path(s))
                shortpath += s + ";"
            elif not self.woff:
                print("Warning: win32fe: Path Not Found: " + pathi)
        shortpath += wfe.get_environment_variable("PATH")
        wfe.set_environment_variable("PATH", shortpath)

    def found_use(self, temp):
        if not self.arg:
            self.arg.append("--help")
            return
        tmp = self.arg.popleft()
        if self.verbose:
            print("Using tool: " + tmp)
        s = wfe.get_short_path(tmp)
        self._use = s if s else tmp     # used if valid path, else as given

    def found_win_l(self, temp):
        self.libprefix = ""
        if self.verbose:
            print("win32fe: Using Windows style -l for compilers, "
                  "-lfoo means foo.lib not libfoo.lib")

    def found_woff(self, temp):
        self.woff = True

    def found_nt4(self, temp):
        if self.verbose:
            print("Warning: win32fe: --NT4 is a deprecated option.")

    def found_verbose(self, temp):
        self.verbose = True

    def found_version(self, temp):
        self.info_action = self.display_version

    def found_help(self, temp):
        self.info_action = self.help

    def found_wait(self, temp):
        pass  # the wait already happened in get_args

    # -- files -----------------------------------------------------------------
    def found_file(self, temp):
        """If the file has a <path> component, verify the directory exists
        and use its short form (mirrors tool::FoundFile)."""
        path = wfe.get_file_path(temp)
        if path is None:
            print("Warning: win32fe: File Not Found: " + temp)
        self.file.append(path if path is not None else temp)

    def is_a_known_tool(self):
        return False
