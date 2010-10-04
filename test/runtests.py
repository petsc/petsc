import sys, os, glob
import unittest

try:
    import petsc4py
except ImportError:
    from distutils.util import get_platform
    plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    os.path.split(__file__)[0]
    path = os.path.join(os.path.split(__file__)[0], os.path.pardir,
                        'build', 'lib' + plat_specifier)
    if not os.path.exists(path) and os.path.exists(path+'-pydebug'):
        path += '-pydebug'
    sys.path.append(path)
    import petsc4py

args=['-malloc',
      '-malloc_debug',
      '-malloc_dump',
      #'-log_summary',
      ]
if '-petsc' in sys.argv:
    idx = sys.argv.index('-petsc')
    args.extend(sys.argv[idx+1:])
    del sys.argv[idx:]
    del idx

petsc4py.init(args)
from petsc4py import PETSc

version = PETSc.Sys.getVersion()
exclude = {}

def test_cases():
    from glob import glob
    directory = os.path.split(__file__)[0]
    pattern = os.path.join(directory, 'test_*.py')
    test_list = []
    for test_file in sorted(glob(pattern)):
        filename = os.path.basename(test_file)
        modulename = os.path.splitext(filename)[0]
        if modulename in exclude:
            if exclude[modulename] is True or \
               exclude[modulename] == version:
                continue
        test = __import__(modulename)
        test_list.append(test)
    return test_list


def runtests(*args, **kargs):
    PETSc.COMM_WORLD.barrier()
    sys.stderr.flush()
    sys.stderr.write("petsc4py imported from '%s'\n" % petsc4py.__path__[0])
    (major, minor, micro), patch = PETSc.Sys.getVersion(patch=True)
    r = PETSc.Sys.getVersionInfo()['release']
    if r: release = 'release'
    else: release = 'development'
    arch = PETSc.__arch__
    sys.stderr.write(
        "using PETSc %d.%d.%d-p%d %s (configuration: '%s')\n" % \
        (major, minor, micro, patch, release, arch) )
    sys.stderr.flush()
    PETSc.COMM_WORLD.barrier()

    for test in test_cases():
        try:
            if PETSc.COMM_WORLD.getRank() == 0:
                sys.stderr.flush()
                sys.stderr.write("\nrunning %s\n" % test.__name__)
                sys.stderr.flush()
            PETSc.COMM_WORLD.barrier()
            unittest.main(test, *args, **kargs)
            PETSc.COMM_WORLD.barrier()
        except SystemExit:
            pass

def runtestsleak(repeats, *args, **kargs):
    import gc
    alltests = test_cases()
    gc.collect()
    for i in range(repeats):
        gc.collect()
        r1 = sys.gettotalrefcount()
        for test in alltests:
            try: unittest.main(test, *args, **kargs)
            except SystemExit: pass
        gc.collect()
        r2 = sys.gettotalrefcount()
        sys.stderr.flush()
        sys.stderr.write('\nREF LEAKS -- before: %d, after: %d, diff: [%d]\n' % (r1, r2, r2-r1))
        sys.stderr.flush()

if __name__ == '__main__':
    runtests()
    if hasattr(sys, 'gettotalrefcount'):
        try:
            from unittest.runner import _WritelnDecorator
        except ImportError:
            from unittest import _WritelnDecorator
        def dummy_write(self,*args): pass
        _WritelnDecorator.write   = dummy_write
        _WritelnDecorator.writeln = dummy_write
        runtestsleak(5)
