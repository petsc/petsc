#!/usr/bin/env python3
"""Test win32fe translation without an installed compiler or librarian."""

import io
import ntpath
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import win32fe_utils as wfe
from win32fe_archiver import Lib
from win32fe_compiler import Cl
from win32fe_frontends import main
from win32fe_tool import Tool


class TranslationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="wfe-test")
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(tmp.name)
        self.cwd = wfe.get_short_path(tmp.name)
        self.native = ntpath if wfe.kernel32() is not None else os.path

    def commands(self, *args, inspect=None):
        calls = []

        def record(tool, argv, skip=None):
            calls.append(argv)
            if inspect:
                inspect(argv)
            return 0

        with patch.object(Tool, "report_filtered", autospec=True, side_effect=record):
            self.assertEqual(main(["win32fe", *args]), 0)
        return calls

    def test_compile(self):
        Path("include dir").mkdir()
        Path("one source.c").touch()
        Path("two.c").touch()
        inc = wfe.get_short_path("include dir")
        macro = '-DNAME="a b"'
        first, second = self.commands("cl", "-c", "one source.c", "two.c", "-o", "out file.o",
                                      "-g", "-O", '-I"include dir"', macro)
        common = ["cl", "-c", "-Z7", "-O2", "-I" + inc, macro, "-nologo"]
        self.assertEqual(first, common + ["-Fo" + self.native.join(self.cwd, "out file.o"),
                                          self.native.join(self.cwd, "one source.c")])
        self.assertEqual(second, common + ["-Fo" + self.native.join(self.cwd, "two.o"),
                                           self.native.join(self.cwd, "two.c")])

    def test_link(self):
        Path("other").mkdir()
        Path("a.o").write_bytes(b"first")
        Path("other/a.o").write_bytes(b"second")
        Path("a.obj").write_bytes(b"keep this object")

        def check_copies(argv):
            objects = [Path(arg) for arg in argv if arg.endswith(".obj")]
            self.assertEqual([p.read_bytes() for p in objects], [b"first", b"second"])

        argv, = self.commands("cl", "a.o", "other/a.o", "-Lother", "-lfoo", "-o", "app name.exe", "-g", inspect=check_copies)
        self.assertIn("libfoo.lib", argv)
        self.assertIn("-libpath:" + wfe.get_short_path("other"), argv)
        self.assertIn("-link", argv)
        self.assertIn("-Fe" + self.native.join(self.cwd, "app name.exe"), argv)
        self.assertFalse(any(Path(arg).exists() for arg in argv if arg.endswith(".obj")))
        self.assertEqual(Path("a.obj").read_bytes(), b"keep this object")

    def test_long_link(self):
        Path("some input.o").touch()

        def check_response(argv):
            path, = [arg[1:] for arg in argv if arg.startswith("@")]
            objects = Path(path).read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(objects), 40)
            self.assertTrue(all(Path(obj.strip('"')).is_file() for obj in objects))

        with patch.object(wfe, "MAX_COMMAND_LINE", 512):
            self.commands("cl", *(["some input.o"] * 40), inspect=check_response)

    def test_help_with_inputs(self):
        Path("source.c").touch()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.commands("cl", "-c", "source.c", "--help"), [["cl", "-?"]])

    def test_tool_selection(self):
        for frontend, driver in (("icl", "icx"), ("ifort", "ifx")):
            argv, = self.commands(frontend, "--use", driver, "--win-l", "-lfoo")
            self.assertEqual(argv, [driver, "-nologo", "foo.lib", "-link"])
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["win32fe", "--help"]), 0)
            self.assertIn("Usage: win32fe", out.getvalue())
            self.assertEqual(main(["win32fe", "--version"]), 0)
            self.assertIn("version", out.getvalue())

    def test_tool_output(self):
        data = b'"caf\xe9"\r\nwarning D4024: unrecognized source file type \'a.o\', object file assumed\r\nlast\xff'
        python = wfe.get_short_path(os.path.realpath(sys.executable))
        arg = 'spaced argument with "quotes" and a trailing \\'
        script = Path("output probe.py")
        script.write_text('import sys\nassert sys.argv[1] == %r\nsys.stdout.buffer.write(%r)\n' % (arg, data), encoding="ascii")
        stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        with redirect_stdout(stream):
            self.assertEqual(Cl().report([python, str(script), arg]), 0, stream.buffer.getvalue())
            stream.flush()
            self.assertEqual(stream.buffer.getvalue(), b'"caf\xe9"\r\nlast\xff')
        stream.close()

    def test_archive_update(self):
        Path("my archive.lib").touch()
        Path("caller file.rsp").touch()
        archive = self.native.join(self.cwd, "my archive.lib")
        user_rsp = "@" + wfe.get_short_path("caller file.rsp")
        members = [letter * 165 + " member.obj" for letter in "abc"]
        generated = []

        def check_response_files(argv):
            cmd, flag, output, before, supplied, after = argv
            self.assertEqual([cmd, flag, output, supplied], ["lib", "-nologo", "-out:" + archive, user_rsp])
            for response, expected in ((before, [archive, members[0]]), (after, members[1:])):
                self.assertTrue(response.startswith("@"))
                path = Path(response[1:])
                self.assertEqual(path.read_text(encoding="utf-8-sig").splitlines(), ['"' + f + '"' for f in expected])
                generated.append(path)

        self.commands("lib", "-a", "my archive.lib", members[0], "@caller file.rsp", *members[1:], inspect=check_response_files)
        self.assertTrue(all(not p.exists() for p in generated))
        self.assertTrue(Path("caller file.rsp").exists())

    def test_archive_members(self):
        Path("my archive.lib").touch()
        archive = self.native.join(self.cwd, "my archive.lib")
        argv, = self.commands("lib", "-l", "my archive.lib", "list file")
        self.assertEqual(argv, ["lib", "-nologo", "-list:list file", archive])
        argv, = self.commands("lib", "-d", "my archive.lib", "one member.obj")
        self.assertEqual(argv, ["lib", "-nologo", archive, "-remove:one member.obj"])

        librarian = Lib()
        librarian.get_args(["win32fe", "lib", "--use", "other-lib", "-x", "my archive.lib"])
        self.assertEqual(librarian.parse(), 0)

        def run_lib(argv):
            if argv[2].startswith("-list:"):
                self.assertEqual([argv[0], argv[1], argv[3]], ["other-lib", "-nologo", archive])
                Path(argv[2][6:]).write_bytes("cour.obj\r\ncœur.obj\r\n".encode("cp1252"))
            return 0

        with patch.object(wfe, "kernel32") as k, patch.object(librarian, "report", side_effect=run_lib) as report:
            k.return_value.GetACP.return_value = 1252
            self.assertEqual(librarian.execute(), 0)
            self.assertEqual([args[0] for args, _ in report.call_args_list[1:]],
                             [["other-lib", "-nologo", "-wx", archive, "-extract:" + name] for name in ("cour.obj", "cœur.obj")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
