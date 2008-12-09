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
exclude = {'test_gc'       : True,
           'test_log'      : (2,3,2),
           }

def test_cases():
    from glob import glob
    directory = os.path.split(__file__)[0]
    pattern = os.path.join(directory, 'test_*.py')
    test_list = []
    for test_file in glob(pattern):
        filename = os.path.basename(test_file)
        modulename = os.path.splitext(filename)[0]
        if modulename in exclude:
            if exclude[modulename] is True or \
               exclude[modulename] == version:
                continue
        test = __import__(modulename)
        test_list.append(test)
    return test_list


PETSc.COMM_WORLD.barrier()
sys.stderr.flush()
sys.stderr.write("petsc4py imported from '%s'\n" % petsc4py.__path__[0])
sys.stderr.flush()
PETSc.COMM_WORLD.barrier()

for test in test_cases():
    try:
        if PETSc.COMM_WORLD.getRank() == 0:
            sys.stderr.flush()
            sys.stderr.write("\nrunning %s\n" % test.__name__)
            sys.stderr.flush()
        PETSc.COMM_WORLD.barrier()
        unittest.main(test)
        PETSc.COMM_WORLD.barrier()
    except SystemExit:
        pass
