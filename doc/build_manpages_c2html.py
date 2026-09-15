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


def _get_arch_tools(petsc_dir, petsc_arch):
    variables = os.path.join(petsc_dir, petsc_arch, 'lib', 'petsc', 'conf', 'petscvariables')
    try:
        with open(variables) as f:
            contents = f.read()
    except OSError as e:
        raise RuntimeError(
            f'PETSC_ARCH={petsc_arch} is not configured; configure it with c2html and sowing before building the documentation'
        ) from e

    executables = {'C2HTML': 'c2html', 'DOCTEXT': 'doctext', 'MAPNAMES': 'mapnames'}
    tools = {}
    for name, executable in executables.items():
        match = re.search(rf'^{name}\s*=\s*(.+?)\s*$', contents, re.MULTILINE)
        if match and os.path.isfile(match.group(1)) and os.access(match.group(1), os.X_OK):
            tools[name] = match.group(1)
        else:
            path = shutil.which(executable)
            if path: tools[name] = path
    missing = [name for name in executables if name not in tools]
    if missing:
        raise RuntimeError(
            f'PETSC_ARCH={petsc_arch} must provide c2html and sowing; not found in petscvariables or PATH: {", ".join(missing)}'
        )
    return tools

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

    petsc_arch = os.environ.get('PETSC_ARCH', 'arch-docs')
    if stage == "pre":
      if petsc_arch == 'arch-docs':
        environment = os.environ.copy()
        environment.pop('PETSC_ARCH', None)
        environment.pop('MAKEFLAGS', None)
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
                   'COPTFLAGS=-O0',
                   'PETSC_ARCH=' + petsc_arch,
                  ]
        if 'PETSCBUIDTARBALL' in os.environ:
          command.append('--download-c2html')
          command.append('--download-sowing')
        else:
          command.append('--with-fc=0')
          if shutil.which('c2html'): command.append('--with-c2html')
          else: command.append('--download-c2html')
          if shutil.which('doctext'): command.append('--with-sowing')
          else: command.append('--download-sowing')

        x = time.clock_gettime(time.CLOCK_REALTIME)
        print('==================================================================')
        print(f'Running {" ".join(command)}', flush=True)
        subprocess.run(command, cwd=petsc_dir, env=environment, check=True)
        print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x), flush=True)
        print('==================================================================')
      tools = _get_arch_tools(petsc_dir, petsc_arch)
      doctext = tools['DOCTEXT']
      print('Using DOCTEXT:', doctext)

      import build_man_pages
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building all manual pages', flush=True)
      docs_roots = _clone_docs_packages(petsc_dir,build_dir)
      build_man_pages.main(petsc_dir,build_dir,doctext,extra_roots=docs_roots)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x), flush=True)
      print('============================================')

      import build_man_examples_links
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building manual page links to tutorials')
      build_man_examples_links.main(petsc_dir,build_dir)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x), flush=True)
      print('============================================')

      import build_man_impls_links
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building manual page links to implementations')
      build_man_impls_links.main(petsc_dir,build_dir)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x), flush=True)
      print('============================================')

      import build_man_index
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building manual page indices')
      build_man_index.main(petsc_dir,build_dir)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x), flush=True)
      print('============================================')
    else:
      tools = _get_arch_tools(petsc_dir, petsc_arch)
      c2html = tools['C2HTML']
      print('Using C2HTML:', c2html)
      mapnames = tools['MAPNAMES']
      print('Using MAPNAMES:', mapnames)
      import build_c2html
      x = time.clock_gettime(time.CLOCK_REALTIME)
      print('============================================')
      print('Building c2html', flush=True)
      build_c2html.main(petsc_dir,build_dir,outdir,c2html,mapnames)
      print("Time: "+str(time.clock_gettime(time.CLOCK_REALTIME) - x), flush=True)
      print('============================================')
