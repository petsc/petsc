#!/usr/bin/env python
""" Configure PETSc and build and place the generated manual pages (as .md files) and html source (as .html files)"""

import os
import errno
import subprocess
import shutil
import argparse
import re

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
                 '--with-regex=0',
                 '--with-mkl_sparse_optimize=0',
                 '--with-mkl_sparse=0',
                 '--with-debugging=0',
                 'COPTFLAS=-O0',
                 '--with-petsc4py',
                 'PETSC_ARCH=' + petsc_arch,
                ]
      if 'PETSCBUIDTARBALL' in os.environ:
        command.append('--download-c2html')
        command.append('--download-sowing')
        c2html = None
        doctext = None
      else:
        command.append('--with-fc=0')
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
      if not doctext:
        with open(os.path.join(petsc_dir,petsc_arch,'lib','petsc','conf','petscvariables')) as f:
          doctext = [line for line in f if line.find('DOCTEXT ') > -1]
          doctext = re.sub('[ ]*DOCTEXT[ ]*=[ ]*','',doctext[0]).strip('\n').strip()
      print('Using DOCTEXT:', doctext)

      import build_man_pages
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building all manual pages')
      build_man_pages.main(petsc_dir,doctext)
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
      build_man_index.main(petsc_dir)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')
    else:
      if not os.path.isfile(os.path.join(petsc_dir, "configure.log")): raise Exception("Expected PETSc configuration not found")
      c2html = shutil.which('c2html')
      if not c2html:
        with open(os.path.join(petsc_dir,petsc_arch,'lib','petsc','conf','petscvariables')) as f:
          c2html = [line for line in f if line.find('C2HTML ') > -1]
          c2html = re.sub('[ ]*C2HTML[ ]*=[ ]*','',c2html[0]).strip('\n').strip()
      print('Using C2HTML:', c2html)
      mapnames = shutil.which('mapnames')
      if not mapnames:
        with open(os.path.join(petsc_dir,petsc_arch,'lib','petsc','conf','petscvariables')) as f:
          mapnames = [line for line in f if line.find('MAPNAMES ') > -1]
          mapnames = re.sub('[ ]*MAPNAMES[ ]*=[ ]*','',mapnames[0]).strip('\n').strip()
      print('Using MAPNAMES:', mapnames)
      import build_c2html
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building c2html')
      build_c2html.main(petsc_dir,outdir,c2html,mapnames)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')
