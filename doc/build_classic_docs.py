#!/usr/bin/env python
""" Configure PETSc and build and place the generated manual pages (as .md files) and html source (as .html files)"""

import os
import errno
import subprocess
import shutil
import argparse

rawhtml = ['include', 'src']
petsc_arch = 'arch-classic-docs'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def main(stage,outdir):
    """ Operations to provide data from the 'classic' PETSc docs system. """
    import time
    petsc_dir = os.path.abspath(os.path.join(THIS_DIR, ".."))  # abspath essential since classic 'html' target uses sed to modify paths from the source to target tree
    if stage == "pre":
      if 'PETSC_ARCH' in os.environ: del os.environ['PETSC_ARCH']
      if 'MAKEFLAGS' in os.environ: del os.environ['MAKEFLAGS']
      command = ['./configure',
                 '--with-coverage-exec=0',
                 '--with-mpi=0',
                 '--with-cxx=0',
                 '--with-syclc=0',
                 '--with-hipc=0',
                 '--with-cudac=0',
                 '--with-x=0',
                 '--with-bison=0',
                 '--with-cmake=0',
                 '--with-pthread=0',
                 '--with-regexp=0',
                 '--with-mkl_sparse_optimize=0',
                 '--with-mkl_sparse=0',
                 '--with-petsc4py',
                 'PETSC_ARCH=' + petsc_arch,
                ]
      if 'PETSCBUIDTARBALL' in os.environ:
        command.append('--download-c2html')
        command.append('--download-sowing')
      else:
        command.append('--with-fc=0')
        import shutil
        c2html = shutil.which('c2html')
        if c2html: command.append('--with-c2html')
        else:  command.append('--download-c2html')
        doctext = shutil.which('doctext')
        if doctext: command.append('--with-sowing')
        else:  command.append('--download-sowing')

      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('==================================================================')
      print('Running configure')
      subprocess.run(command, cwd=petsc_dir, check=True)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('==================================================================')

      loc = os.getcwd()
      command = ['make', 'allmanpages',
                 'PETSC_DIR=%s' % petsc_dir,
                 'PETSC_ARCH=%s' % petsc_arch,
                 'HTMLMAP=%s' % os.path.join(os.getcwd(),'manualpages','htmlmap'),
                 'LOC=%s' % loc]
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('make allmanpages')
      subprocess.run(command, cwd=petsc_dir, check=True)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')

      import build_man_examples_links
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building manual page links to tutorials')
      build_man_examples_links.main(petsc_dir)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')

      import build_man_impls_links
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building manual page links to implementations')
      build_man_impls_links.main(petsc_dir)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')

      import build_man_index
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building manual page indices')
      build_man_index.main(petsc_dir,loc)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')
    else:
      if not os.path.isfile(os.path.join(petsc_dir, "configure.log")): raise Exception("Expected PETSc configuration not found")
      loc = outdir
      command = ['make', 'c2html',
                 'PETSC_DIR=%s' % petsc_dir,
                 'PETSC_ARCH=%s' % petsc_arch,
                 'HTMLMAP=%s' % os.path.join(os.getcwd(),'manualpages','htmlmap'),
                 'LOC=%s' % loc]
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('make c2html')
      subprocess.run(command, cwd=petsc_dir, check=True)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')
