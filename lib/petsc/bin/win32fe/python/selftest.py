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


if __name__ == "__main__":
    unittest.main(verbosity=2)
