"""Low-level helpers for the Python port of win32fe.

Mirrors src/utils.cpp, src/win32fepath.c, src/mycygpath.cpp and
src/outpipe.cpp.  Designed to run under Cygwin Python.  When kernel32 is
unavailable, as when selftest.py runs on Linux, the Win32 calls use os and
subprocess instead, so the argument translation can be tested without Windows.
"""

import ctypes
import ntpath
import os
import sys

BIG_PATH = 32768  # buffer size for the Win32 path APIs (long paths)
MAX_COMMAND_LINE = 32767  # CreateProcessW limit in UTF-16 code units, including the terminator

# --- Win32 constants -------------------------------------------------------
STARTF_USESTDHANDLES = 0x00000100
STD_INPUT_HANDLE = -10
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
DELETE_ACCESS = 0x00010000
STANDARD_RIGHTS_ALL = 0x001F0000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
CREATE_ALWAYS = 2
FILE_ATTRIBUTE_NORMAL = 0x00000080
INFINITE = 0xFFFFFFFF
CW_SYNC_WINENV = 32  # cygwin_getinfo_types in /usr/include/sys/cygwin.h
CCP_POSIX_TO_WIN_W = 1  # cygwin_conv_path_t in /usr/include/sys/cygwin.h
# How ctypes represents (HANDLE)-1, on any bitness:
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("nLength", ctypes.c_uint32),
                ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle", ctypes.c_int)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32),
                ("lpReserved", ctypes.c_wchar_p),
                ("lpDesktop", ctypes.c_wchar_p),
                ("lpTitle", ctypes.c_wchar_p),
                ("dwX", ctypes.c_uint32), ("dwY", ctypes.c_uint32),
                ("dwXSize", ctypes.c_uint32), ("dwYSize", ctypes.c_uint32),
                ("dwXCountChars", ctypes.c_uint32), ("dwYCountChars", ctypes.c_uint32),
                ("dwFillAttribute", ctypes.c_uint32), ("dwFlags", ctypes.c_uint32),
                ("wShowWindow", ctypes.c_uint16), ("cbReserved2", ctypes.c_uint16),
                ("lpReserved2", ctypes.c_void_p),
                ("hStdInput", ctypes.c_void_p),
                ("hStdOutput", ctypes.c_void_p),
                ("hStdError", ctypes.c_void_p)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", ctypes.c_void_p), ("hThread", ctypes.c_void_p),
                ("dwProcessId", ctypes.c_uint32), ("dwThreadId", ctypes.c_uint32)]


_kernel32 = None
_kernel32_loaded = False


def _proto(k, name, restype, argtypes):
    f = getattr(k, name)
    f.restype = restype
    f.argtypes = argtypes


def kernel32():
    """Return kernel32 with prototypes set, or None without it (selftest.py on Linux)."""
    global _kernel32, _kernel32_loaded
    if _kernel32_loaded:
        return _kernel32
    _kernel32_loaded = True
    try:
        # Cygwin Python has no WinDLL. 64-bit Windows has one calling convention, so CDLL is safe
        if hasattr(ctypes, "WinDLL"):
            k = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        else:
            k = ctypes.CDLL("kernel32.dll", use_last_error=True)
        _proto(k, "GetShortPathNameW", ctypes.c_uint,
               [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint])
        _proto(k, "GetLongPathNameW", ctypes.c_uint,
               [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint])
        _proto(k, "GetCurrentProcessId", ctypes.c_uint32, [])
        _proto(k, "GetACP", ctypes.c_uint, [])
        _proto(k, "GetTempPathW", ctypes.c_uint, [ctypes.c_uint, ctypes.c_wchar_p])
        _proto(k, "GetTempFileNameW", ctypes.c_uint,
               [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_wchar_p])
        _proto(k, "GetStdHandle", ctypes.c_void_p, [ctypes.c_int])
        _proto(k, "CreateFileW", ctypes.c_void_p,
               [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p])
        _proto(k, "CloseHandle", ctypes.c_int, [ctypes.c_void_p])
        _proto(k, "DeleteFileW", ctypes.c_int, [ctypes.c_wchar_p])
        _proto(k, "CopyFileW", ctypes.c_int, [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int])
        _proto(k, "GetEnvironmentVariableW", ctypes.c_uint32,
               [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32])
        _proto(k, "SetEnvironmentVariableW", ctypes.c_int, [ctypes.c_wchar_p, ctypes.c_wchar_p])
        _proto(k, "CreateProcessW", ctypes.c_int,
               [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_wchar_p,
                ctypes.c_void_p, ctypes.c_void_p])
        _proto(k, "WaitForSingleObject", ctypes.c_uint32, [ctypes.c_void_p, ctypes.c_uint32])
        _proto(k, "GetExitCodeProcess", ctypes.c_int,
               [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)])
        _kernel32 = k
    except (OSError, AttributeError):
        _kernel32 = None
    if _kernel32 is not None and sys.platform == "cygwin":
        # Give native tools the Cygwin environment, for example INCLUDE and LIB
        ctypes.CDLL("cygwin1.dll").cygwin_internal(CW_SYNC_WINENV)
    return _kernel32


# --------------------------------------------------------------------------
# POSIX <-> Win32 path conversion (mirrors mycygpath.cpp and the
# cygwin_conv_to_full_win32_path part of win32fepath.c)
# --------------------------------------------------------------------------

_cygwin_conv = None


def _cygwin_conv_path(path):
    """Absolute Windows form of path from cygwin_conv_path() in cygwin1.dll.
    This is the conversion that 'cygpath -aw' does, without a process."""
    global _cygwin_conv
    if _cygwin_conv is None:
        _cygwin_conv = ctypes.CDLL("cygwin1.dll").cygwin_conv_path
        _cygwin_conv.restype = ctypes.c_ssize_t
        _cygwin_conv.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    src = os.fsencode(path)
    size = _cygwin_conv(CCP_POSIX_TO_WIN_W, src, None, 0)   # bytes, with the terminator
    if size <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(size // ctypes.sizeof(ctypes.c_wchar))
    return buf.value if _cygwin_conv(CCP_POSIX_TO_WIN_W, src, buf, size) == 0 else ""


def _run_cygpath(args, path):
    import subprocess
    try:
        r = subprocess.run(["cygpath", *args, path],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True)
        if r.returncode == 0:
            return r.stdout.strip()
    except OSError:
        pass
    return ""


def _strip_long_path_prefix(path):
    """Remove the \\\\?\\ prefix that Win32 adds to paths longer than MAX_PATH."""
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[8:]
    return path[4:] if path.startswith("\\\\?\\") else path


def cygwin_conv_to_full_win32_path(p):
    """Mirror of my_cygwin_conv_to_full_win32_path: 'cygpath -aw <p>'."""
    out = _cygwin_conv_path(p) if sys.platform == "cygwin" else _run_cygpath(["-aw"], p)
    if out:
        return out
    if len(p) > 2 and ("a" <= p[0] <= "z" or "A" <= p[0] <= "Z") and p[1] == ":" \
            and p[2] in "\\/":       # already a Windows path
        return p
    if os.name == "nt":               # native Windows Python
        return os.path.abspath(p)
    return p


def unquote_path(path):
    if len(path) >= 2 and path[0] in "\"'" and path[-1] == path[0]:
        return path[1:-1]
    return path


def get_short_path(path):
    """Mirror of win32fe::GetShortPath (win32fe_GetShortPath_default):
    returns the short (8.3) Windows form of path, or None if the path does
    not exist (GetShortPathName fails on non-existent paths)."""
    path = unquote_path(path)
    win = cygwin_conv_to_full_win32_path(path)
    k = kernel32()
    if k is not None:
        buf = ctypes.create_unicode_buffer(BIG_PATH)
        if k.GetShortPathNameW(win, buf, BIG_PATH) > 0:
            return _strip_long_path_prefix(buf.value)
        return None
    # Without kernel32 (selftest.py on Linux): existence check only, keep converted path.
    for cand in (path, path.replace("\\", "/")):
        if os.path.exists(cand):
            return win
    return None


def get_file_path(path, absolute=False):
    """Use the short directory name even when the file does not yet exist."""
    path = unquote_path(path)
    directory, name = ntpath.split(path)
    if not directory and not absolute:
        return name
    directory = get_short_path(directory or os.getcwd())
    if directory is None:
        return None
    return (ntpath if kernel32() is not None else os.path).join(directory, name)


def get_long_path(shortpath):
    """Mirror of win32fe::GetLongPath."""
    k = kernel32()
    if k is not None:
        buf = ctypes.create_unicode_buffer(BIG_PATH)
        if k.GetLongPathNameW(shortpath, buf, BIG_PATH) > 0:
            return buf.value
    return shortpath


def get_unique_temp_file_name():
    """Mirror of win32fe::GetUniqueTempFileName (GetTempPath+GetTempFileName)."""
    k = kernel32()
    if k is not None:
        dirbuf = ctypes.create_unicode_buffer(BIG_PATH)
        fbuf = ctypes.create_unicode_buffer(BIG_PATH)
        if k.GetTempPathW(BIG_PATH, dirbuf) > 0 and \
                k.GetTempFileNameW(dirbuf.value, "wfe", 0, fbuf) != 0:
            return fbuf.value
    import tempfile
    with tempfile.NamedTemporaryFile(prefix="wfe", delete=False) as f:
        name = f.name
    return cygwin_conv_to_full_win32_path(name)


def create_file_inheritable(filename):
    """Mirror of win32fe::CreateFile (inheritable handle for child stdio)."""
    k = kernel32()
    if k is None:
        return None
    sa = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), None, 1)
    h = k.CreateFileW(filename,
                      GENERIC_READ | GENERIC_WRITE | DELETE_ACCESS | STANDARD_RIGHTS_ALL,
                      FILE_SHARE_READ | FILE_SHARE_WRITE,
                      ctypes.byref(sa), CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
    if h is None or h == INVALID_HANDLE_VALUE:
        return None
    return h


# --------------------------------------------------------------------------
# File / environment helpers (mirrors the tail of utils.cpp)
# --------------------------------------------------------------------------

def delete_file(filename, verbose=False):
    if verbose:
        print("del " + filename)
    k = kernel32()
    if k is not None:
        return bool(k.DeleteFileW(filename))
    try:
        os.remove(filename)
        return True
    except OSError:
        return False


def copy_file(src, dst, verbose=False):
    if verbose:
        print("win32fe::CopyFile copying %s to %s" % (src, dst))
    k = kernel32()
    if k is not None:
        return bool(k.CopyFileW(src, dst, 0))
    import shutil
    shutil.copy(src, dst)
    return True


def get_environment_variable(name):
    """Mirror of win32fe::GetEnvironmentVariable (the Windows environment)."""
    k = kernel32()
    if k is not None:
        buf = ctypes.create_unicode_buffer(BIG_PATH)
        n = k.GetEnvironmentVariableW(name, buf, BIG_PATH)
        return buf.value if 0 < n < BIG_PATH else ""
    return os.environ.get(name, "")


def set_environment_variable(name, value):
    """Mirror of win32fe::SetEnvironmentVariable: native tools inherit it."""
    k = kernel32()
    if k is not None:
        return bool(k.SetEnvironmentVariableW(name, value))
    os.environ[name] = value
    return True


# --------------------------------------------------------------------------
# OutputPipe: serialize argv for CreateProcessW and capture the child's bytes.
# --------------------------------------------------------------------------

def list2cmdline(argv):
    """Quote argv like subprocess.list2cmdline(), without the cost of importing subprocess."""
    parts = []
    for arg in argv:
        quote = not arg or " " in arg or "\t" in arg
        out, backslashes = [], 0
        for c in arg:
            if c == "\\":
                backslashes += 1
                continue
            out.append("\\" * (2 * backslashes + 1) + '"' if c == '"' else "\\" * backslashes + c)
            backslashes = 0
        out.append("\\" * (2 * backslashes if quote else backslashes))
        parts.append('"' + "".join(out) + '"' if quote else "".join(out))
    return " ".join(parts)


def command_line_size(argv):
    return len(list2cmdline(argv).encode("utf-16-le", "surrogatepass")) // 2 + 1


def response_file_args(directory, args, option=None):
    """Pack adjacent inputs in response files without nesting caller-supplied @files."""
    from itertools import groupby
    for i, (supplied, group) in enumerate(groupby(args, key=lambda arg: arg.startswith("@"))):
        if supplied:
            yield from group
        else:
            path = os.path.join(directory, "%d.rsp" % i)
            with open(path, "w", encoding="utf-8" if option else "utf-8-sig", newline="\r\n") as rsp:
                for arg in group:
                    rsp.write(list2cmdline([arg]) + "\n")
            native = cygwin_conv_to_full_win32_path(path)
            if option:
                yield option
                yield native
            else:
                yield "@" + native


class OutputPipe:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.outfilename = get_unique_temp_file_name()
        self.outfile = create_file_inheritable(self.outfilename)
        self.si = STARTUPINFOW()
        self.si.cb = ctypes.sizeof(STARTUPINFOW)
        self.si.dwFlags = STARTF_USESTDHANDLES
        k = kernel32()
        self.si.hStdInput = k.GetStdHandle(STD_INPUT_HANDLE) if k is not None else None
        self.si.hStdOutput = self.outfile
        self.si.hStdError = self.outfile
        self.pi = PROCESS_INFORMATION()
        self._proc = None      # subprocess used without kernel32 (selftest.py on Linux)
        self._reader = None    # open(outfilename) once the child has exited

    def start(self, argv):
        cmdline = list2cmdline(argv)
        if self.verbose:
            print(cmdline, flush=True)
        k = kernel32()
        if k is not None:
            buf = ctypes.create_unicode_buffer(cmdline)
            ok = k.CreateProcessW(None, ctypes.cast(buf, ctypes.c_void_p),
                                  None, None, 1, 0, None, None,
                                  ctypes.byref(self.si), ctypes.byref(self.pi))
            started = bool(ok)
            if started:
                k.CloseHandle(self.pi.hThread)   # only hProcess is needed
                self.pi.hThread = None
        else:
            # Without kernel32 (selftest.py on Linux): plain subprocess
            import subprocess
            try:
                with open(self.outfilename, "wb") as f:
                    self._proc = subprocess.Popen(argv, stdout=f,
                                                  stderr=subprocess.STDOUT)
                started = True
            except OSError:
                started = False
        if not started:
            print("win32fe: failed to execute: " + cmdline, file=sys.stderr)
        return started

    def get_exit_code(self):
        k = kernel32()
        if k is not None and self.pi.hProcess:
            k.WaitForSingleObject(self.pi.hProcess, INFINITE)
            code = ctypes.c_uint32(0)
            k.GetExitCodeProcess(self.pi.hProcess, ctypes.byref(code))
            exit_code = code.value
            k.CloseHandle(self.pi.hProcess)
            self.pi.hProcess = None
            if self.outfile is not None:
                k.CloseHandle(self.outfile)
                self.outfile = None
        elif self._proc is not None:
            exit_code = self._proc.wait()
        else:
            exit_code = 1
        self._reader = open(self.outfilename, "rb")
        return exit_code

    def execute(self, argv):
        """Run start(), wait for the child, and return its exit code."""
        if not self.start(argv):
            return 1
        return self.get_exit_code()

    def get_line(self):
        return self._reader.readline() if self._reader is not None else b""

    def output_line(self):
        sys.stdout.flush()
        sys.stdout.buffer.write(self.get_line())

    def output_all(self, skip=None):
        sys.stdout.flush()
        for line in iter(self.get_line, b""):
            if skip is None or not skip(line):
                sys.stdout.buffer.write(line)

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        k = kernel32()
        if k is not None:
            if self.outfile is not None:
                k.CloseHandle(self.outfile)
                self.outfile = None
            if self.pi.hProcess:
                k.CloseHandle(self.pi.hProcess)
                self.pi.hProcess = None
        if self.outfilename:
            delete_file(self.outfilename, self.verbose)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
