"""Archiver (librarian) front-ends for the Python port of win32fe (mirrors
archiverfe.cpp, libfe.cpp).
"""

import ntpath
import os
import sys
from collections import deque

import win32fe_utils as wfe
from win32fe_tool import Tool

ARCHIVER_HELP = """\
For archivers:
  win32fe will map the following <tool options> to their native options:
    Use only one of the following options per invocation of win32fe.
    -a <archivename> <{object list}>
         Archive the list of objects.
         The first filename listed will be taken as the archivename.
    -d <archivename> <{object list}>
         Delete the list of objects from the archive.
    -l <archivename> <destination>:
         List objects in the library to a file.
         If no <destination> is specified, stdout is assumed.
    -x <archivename> <{object list}>
         Extract the list of objects from the archive.
         If no list is provided, all objects will be extracted.
    @<cmdfilename>
         Specify command file [with native commands listed in file] to archiver.
Note: Long lists of members are passed to LIB in response files.

===========================================================================
"""

LIB_HELP = """\
lib specific help:
  *   win32fe lib -a will check to see if the library exists, and will
      modify the arguments to lib accordingly; therefore, specifying
      -out:<libname> is unnecessary and may lead to unexpected results.
  *   Use the win32fe options -l, -x, and -d instead of their
      lib counterparts -list:, -extract:, and -remove:, respectively.
  Ex: win32fe lib -a libfoo.lib foo.o bar.o

============================================================================
"""


class Archiver(Tool):
    """Base class for archiver front-ends (mirrors archiverfe.h)."""

    def __init__(self):
        super().__init__()
        self.aoptions = {"-a": self.archive, "-d": self.delete,
                         "-l": self.list, "-x": self.extract}
        self.action = self.help         # archive/delete/list/extract
        self.archivearg = deque()       # translated archiver command args
        self.archive_name = ""
        self.archive_exists = False

    def parse(self):
        if super().parse():
            return -1
        if self.arg:
            self.archivearg.appendleft(self.arg.popleft())  # the tool command
        while self.arg:
            temp = self.arg.popleft()
            if temp.startswith("@"):
                self.found_cmdfile(temp)
            elif not temp.startswith("-"):
                self.found_file(temp)
            else:
                h = self.aoptions.get(temp)
                if h is None:
                    self.found_flag(temp)
                else:
                    self.action = h
        return 0

    def archive(self):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

    def extract(self):
        raise NotImplementedError

    def help(self):
        super().help()
        print(ARCHIVER_HELP)
        return 0

    def found_file(self, temp):
        if self.archive_name:
            super().found_file(temp)
        else:
            self.found_archive(temp)

    def found_archive(self, temp):
        self.archive_exists = wfe.get_short_path(temp) is not None
        self.archive_name = wfe.get_file_path(temp, absolute=True)
        if self.archive_name is None:
            if not self.woff:
                print("Warning: win32fe: Directory not found: " + ntpath.dirname(temp))
            self.archive_name = temp

    def found_cmdfile(self, temp):
        s = wfe.get_short_path(temp[1:])
        if s:
            self.file.append("@" + s)
        elif not self.woff:
            print("Warning: win32fe: Command File Not Found: " + temp[1:])

    def found_flag(self, temp):
        self.archivearg.append(temp)

    def display_version(self):
        super().display_version()
        with wfe.OutputPipe(self.verbose) as out:
            if out.start([self.archivearg[0]]):
                out.get_exit_code()
                out.output_line()
                return 0
            return 1

    def is_a_known_tool(self):
        return True


class Lib(Archiver):
    """Microsoft Library Manager front-end (mirrors libfe.cpp)."""

    def __init__(self):
        super().__init__()
        self.action = self.archive

    def parse(self):
        if super().parse():
            return -1
        if not self.verbose:
            self.archivearg.append("-nologo")
        return 0

    def archive(self):
        header = [*self.archivearg, "-out:" + self.archive_name]
        members = [self.archive_name, *self.file] if self.archive_exists else list(self.file)
        if not members:
            return 0
        if wfe.command_line_size(header + members) <= 512:
            return self.report(header + members)

        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix="wfe") as directory:
            prefix = header
            batch = []
            size = wfe.command_line_size(header)
            limit = wfe.MAX_COMMAND_LINE - 64  # LIB.EXE adds /lib when it starts LINK.EXE
            for arg in wfe.response_file_args(directory, members):
                extra = wfe.command_line_size([arg])  # includes the space before this argument
                if batch and size + extra > limit:
                    ierr = self.report(header + batch)
                    if ierr:
                        return ierr
                    # An earlier batch creates the archive only if it holds members
                    header = prefix + ([self.archive_name] if wfe.get_short_path(self.archive_name) else [])
                    size = wfe.command_line_size(header)
                    batch = []
                if size + extra > limit:
                    print("Error: win32fe: LIB command exceeds the Windows command-line limit: " + arg, file=sys.stderr)
                    return 1
                batch.append(arg)
                size += extra
            return self.report(header + batch)

    def delete(self):
        return self.edit_members("-remove:")

    def edit_members(self, option):
        for member in self.file:
            ierr = self.report([*self.archivearg, self.archive_name, option + member])
            if ierr:
                return ierr
        return 0

    def list(self):
        if len(self.file) > 1:
            print("Warning: win32fe: too many files passed with -l")
            return -1
        option = "-list:" + self.file[0] if self.file else "-list"
        return self.report([*self.archivearg, option, self.archive_name])

    def get_file_list_from_archive(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix="wfe") as directory:
            path = os.path.join(directory, "members.txt")
            ierr = self.report([self.archivearg[0], "-nologo", "-list:" + wfe.cygwin_conv_to_full_win32_path(path), self.archive_name])
            if ierr:
                return ierr
            # LIB uses the Windows ANSI code page for a named listing and the console code page for stdout.
            k = wfe.kernel32()
            encoding = "cp%d" % k.GetACP() if k is not None else sys.getfilesystemencoding()
            try:
                with open(path, encoding=encoding) as listing:
                    self.file.extend(line.rstrip("\n") for line in listing)
            except (OSError, UnicodeDecodeError) as err:
                print("Error: win32fe: cannot read archive member list: %s" % err, file=sys.stderr)
                return 1
        return 0

    def extract(self):
        if not self.file:
            ierr = self.get_file_list_from_archive()
            if ierr:
                print("Error: win32fe: Failed to get list of files from archive.")
                return ierr
            if len(set(self.file)) != len(self.file):
                print("Error: win32fe: LIB listed duplicate member names. Specify the names to extract with -x.", file=sys.stderr)
                return 1
            self.archivearg.append("-wx")  # LIB otherwise returns success for a missing member.
        return self.edit_members("-extract:")

    def help(self):
        super().help()
        print(LIB_HELP)
        return self.report([self.archivearg[0], "-?"])

    def found_verbose(self, temp):
        super().found_verbose(temp)
        self.archivearg.append("-verbose")
