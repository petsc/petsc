#!/usr/bin/env python3
""" Configure PETSc and build and place the generated manual pages (as .md files) and html source (as .html files)"""

import os
import errno
import subprocess
import shutil
import argparse
import re
import ast
import glob

rawhtml = ['include', 'src']
petsc_arch = 'arch-docs'

def _provides_docs_packages(petsc_dir):
    """Return list of (name, giturl, gitcommit, docsDirs) for packages with providesDocs=1.

    Parsed statically from config/BuildSystem/config/packages/*.py (no instantiation) since
    the docs configure never processes these packages (they sit behind the MPI/Fortran/BLAS
    dependency wall)."""
    out = []
    pkgdir = os.path.join(petsc_dir, 'config', 'BuildSystem', 'config', 'packages')
    for path in glob.glob(os.path.join(pkgdir, '*.py')):
        with open(path) as f:
            src = f.read()
        if 'providesDocs' not in src: continue
        vals = {}     # attribute name -> literal value
        giturl = None # git URL from the self.download list; the git entry looks like 'git://https://github.com/.../PFLARE'
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)): continue
            attr = node.targets[0].attr
            try:
                vals[attr] = ast.literal_eval(node.value)
            except Exception:
                pass
            if attr == 'download' and isinstance(node.value, ast.List):
                # self.download mixes literals with concatenations (e.g. '...'+self.gitcommit+'...'),
                # so literal_eval of the whole list fails; evaluate elements individually to find the git URL
                for elt in node.value.elts:
                    try:
                        d = ast.literal_eval(elt)
                    except Exception:
                        continue
                    if isinstance(d, str) and d.startswith('git://'):
                        giturl = d[len('git://'):]
                        break
        if not vals.get('providesDocs'): continue
        out.append((os.path.splitext(os.path.basename(path))[0], giturl, vals.get('gitcommit'), vals.get('docsDirs', ['src'])))
    return out

def _clone_docs_packages(petsc_dir, build_dir):
    """Clone each providesDocs package at its pinned commit into a scratch dir under build_dir
    and return (repo_root, docs_dir) pairs to scan; repo_root is passed to the scanner as the
    base so source-location paths in the generated pages read as 'src/...'/'include/...' relative
    to the package. A plain (non-shallow) clone is used because the pinned commit is checked out
    afterwards and may not be the branch tip. Only the docsDirs are kept in the clone:
    the scratch dir lives inside the Sphinx source tree, so keeping the whole repository would
    make Sphinx render the package's own README/docs Markdown as website pages. Defensive: a
    failed clone (offline) only warns so the core PETSc docs still build."""
    roots = []
    base = os.path.join(build_dir, 'packages-docs')
    for name, giturl, commit, docsDirs in _provides_docs_packages(petsc_dir):
        if not giturl:
            print('Skipping docs clone for %s: no git URL' % name)
            continue
        dest = os.path.join(base, name)
        try:
            if not os.path.isdir(dest):
                os.makedirs(base, exist_ok=True)
                subprocess.run(['git', 'clone', giturl, dest], check=True)
            else:
                # a cached clone from a previous build may predate a bump of self.gitcommit,
                # so fetch before checkout or the pinned commit may not be present locally
                subprocess.run(['git', '-C', dest, 'fetch', '-q', 'origin'], check=True)
            if commit:
                subprocess.run(['git', '-C', dest, 'checkout', '-q', commit], check=True)
            # keep only the docsDirs (and .git, so a cached clone can be re-checked-out); a
            # checkout to a new commit restores the full tree, so prune the rest on every pass
            for entry in os.listdir(dest):
                if entry in docsDirs or entry == '.git': continue
                path = os.path.join(dest, entry)
                if os.path.isdir(path): shutil.rmtree(path)
                else: os.remove(path)
            roots += [(dest, os.path.join(dest, d)) for d in docsDirs if os.path.isdir(os.path.join(dest, d))]
        except Exception as e:
            print('WARNING: could not clone docs for %s (%s); its manual pages will be missing' % (name, e))
    return roots

def main(stage,petsc_dir,build_dir,outdir):
    """ Operations to provide data for PETSc manual pages and c2html files. """
    import time

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
                 '--with-mkl_sparse_optimize=0',
                 '--with-mkl_sparse=0',
                 '--with-debugging=0',
                 '--download-sowing=1',
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
      docs_roots = _clone_docs_packages(petsc_dir,build_dir)
      build_man_pages.main(petsc_dir,build_dir,doctext,extra_roots=docs_roots)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')

      import build_man_examples_links
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building manual page links to tutorials')
      build_man_examples_links.main(petsc_dir,build_dir)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')

      import build_man_impls_links
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building manual page links to implementations')
      build_man_impls_links.main(petsc_dir,build_dir)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')

      import build_man_index
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building manual page indices')
      build_man_index.main(petsc_dir,build_dir)
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
      build_c2html.main(petsc_dir,build_dir,outdir,c2html,mapnames)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x))
      print('============================================')
