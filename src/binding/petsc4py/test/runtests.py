import sys, os
import optparse
import unittest

def getoptionparser():
    parser = optparse.OptionParser()

    parser.add_option("-q", "--quiet",
                      action="store_const", const=0, dest="verbose", default=1,
                      help="do not print status messages to stdout")
    parser.add_option("-v", "--verbose",
                      action="store_const", const=2, dest="verbose", default=1,
                      help="print status messages to stdout")
    parser.add_option("-i", "--include", type="string",
                      action="append",  dest="include", default=[],
                      help="include tests matching PATTERN", metavar="PATTERN")
    parser.add_option("-e", "--exclude", type="string",
                      action="append", dest="exclude", default=[],
                      help="exclude tests matching PATTERN", metavar="PATTERN")
    parser.add_option("-k", "--pattern", type="string",
                      action="append", dest="patterns", default=[],
                      help="only run tests which match the given substring")
    parser.add_option("-f", "--failfast",
                      action="store_true", dest="failfast", default=False,
                      help="Stop on first failure")
    parser.add_option("--no-builddir",
                      action="store_false", dest="builddir", default=True,
                      help="disable testing from build directory")
    parser.add_option("--path", type="string",
                      action="append", dest="path", default=[],
                      help="prepend PATH to sys.path", metavar="PATH")
    parser.add_option("--refleaks", type="int",
                      action="store", dest="repeats", default=3,
                      help="run tests REPEAT times in a loop to catch leaks",
                      metavar="REPEAT")
    parser.add_option("--arch", type="string",
                      action="store", dest="arch", default=None,
                      help="use PETSC_ARCH",
                      metavar="PETSC_ARCH")
    parser.add_option("-s","--summary",
                      action="store_true", dest="summary", default=0,
                      help="print PETSc log summary")
    return parser

def getbuilddir():
    from distutils.util import get_platform
    s = os.path.join("build", "lib.%s-%.3s" % (get_platform(), sys.version))
    if hasattr(sys, 'gettotalrefcount'): s += '-pydebug'
    return s

def setup_python(options):
    rootdir = os.path.dirname(os.path.dirname(__file__))
    builddir = os.path.join(rootdir, getbuilddir())
    if options.builddir and os.path.exists(builddir):
        sys.path.insert(0, builddir)
    if options.path:
        path = options.path[:]
        path.reverse()
        for p in path:
            sys.path.insert(0, p)

def setup_unittest(options):
    from unittest import TestSuite
    try:
        from unittest.runner import _WritelnDecorator
    except ImportError:
        from unittest import _WritelnDecorator
    #
    writeln_orig = _WritelnDecorator.writeln
    def writeln(self, message=''):
        try: self.stream.flush()
        except: pass
        writeln_orig(self, message)
        try: self.stream.flush()
        except: pass
    _WritelnDecorator.writeln = writeln

def import_package(options, pkgname):
    args = [
        sys.argv[0],
        '-malloc',
        '-malloc_debug',
        '-malloc_dump',
    ]
    if options.summary:
        args.append('-log_view')
    package = __import__(pkgname)
    package.init(args, arch=options.arch)
    return package

def getprocessorinfo():
    try:
        name = os.uname()[1]
    except:
        import platform
        name = platform.uname()[1]
    from petsc4py.PETSc import COMM_WORLD
    rank = COMM_WORLD.getRank()
    return (rank, name)

def getlibraryinfo():
    from petsc4py import PETSc
    (major, minor, micro) = PETSc.Sys.getVersion()
    r = PETSc.Sys.getVersionInfo()['release']
    if r: release = 'release'
    else: release = 'development'
    arch = PETSc.__arch__
    return ("PETSc %d.%d.%d %s (conf: '%s')"
            % (major, minor, micro, release, arch) )
    
def getpythoninfo():
    x, y = sys.version_info[:2]
    return ("Python %d.%d (%s)" % (x, y, sys.executable))

def getpackageinfo(pkg):
    return ("%s %s (%s)" % (pkg.__name__,
                            pkg.__version__,
                            pkg.__path__[0]))

def writeln(message='', endl='\n'):
    from petsc4py.PETSc import Sys
    Sys.syncPrint(message, endl=endl, flush=True)

def print_banner(options, package):
    r, n = getprocessorinfo()
    fmt = "[%d@%s] %s"
    if options.verbose:
        writeln(fmt % (r, n, getpythoninfo()))
        writeln(fmt % (r, n, getlibraryinfo()))
        writeln(fmt % (r, n, getpackageinfo(package)))

def load_tests(options, args):
    from glob import glob
    import re
    testsuitedir = os.path.dirname(__file__)
    sys.path.insert(0, testsuitedir)
    pattern = 'test_*.py'
    wildcard = os.path.join(testsuitedir, pattern)
    testfiles = glob(wildcard)
    testfiles.sort()
    testsuite = unittest.TestSuite()
    testloader = unittest.TestLoader()
    if options.patterns:
        testloader.testNamePatterns = [
            ('*%s*' % p) if ('*' not in p) else p
            for p in options.patterns]
    include = exclude = None
    if options.include:
        include = re.compile('|'.join(options.include)).search
    if options.exclude:
        exclude = re.compile('|'.join(options.exclude)).search
    for testfile in testfiles:
        filename = os.path.basename(testfile)
        testname = os.path.splitext(filename)[0]
        if ((exclude and exclude(testname)) or
            (include and not include(testname))):
            continue
        module = __import__(testname)
        for arg in args:
            try:
                cases = testloader.loadTestsFromNames((arg,), module)
                testsuite.addTests(cases)
            except AttributeError:
                pass
        if not args:
            cases = testloader.loadTestsFromModule(module)
            testsuite.addTests(cases)
    return testsuite

def run_tests(options, testsuite, runner=None):
    if runner is None:
        runner = unittest.TextTestRunner(verbosity=options.verbose)
        runner.failfast = options.failfast
    result = runner.run(testsuite)
    return result.wasSuccessful()

def test_refleaks(options, args):
    from sys import gettotalrefcount
    from gc import collect
    from copy import deepcopy
    testsuite = load_tests(options, args)
    class EmptyIO(object):
        def write(self, *args):
            pass
    runner = unittest.TextTestRunner(stream=EmptyIO(), verbosity=0)
    rank, name = getprocessorinfo()
    r1 = r2 = 0
    repeats = options.repeats
    while repeats:
        collect()
        r1 = gettotalrefcount()
        run_tests(options, deepcopy(testsuite), runner)
        collect()
        r2 = gettotalrefcount()
        leaks = r2-r1
        if leaks and repeats < options.repeats:
            writeln('[%d@%s] refleaks:  (%d - %d) --> %d'
                    % (rank, name, r2, r1, leaks))
        repeats -= 1

def abort(code=1):
    os.abort()

def shutdown(success):
    pass

def main(args=None):
    pkgname = 'petsc4py'
    parser = getoptionparser()
    (options, args) = parser.parse_args(args)
    setup_python(options)
    setup_unittest(options)
    package = import_package(options, pkgname)
    print_banner(options, package)
    testsuite = load_tests(options, args)
    success = run_tests(options, testsuite)
    if not success and options.failfast: abort()
    if success and hasattr(sys, 'gettotalrefcount'):
        test_refleaks(options, args)
    shutdown(success)
    return not success

if __name__ == '__main__':
    import sys
    sys.dont_write_bytecode = True
    sys.exit(main())
