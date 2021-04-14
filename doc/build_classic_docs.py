#!/usr/bin/env python
""" Configure PETSc and build and place still-required classic docs"""

import os
import errno
import subprocess
import shutil


def main():
    """ Operations to provide data from the 'classic' PETSc docs system. """
    petsc_dir = os.path.abspath(os.path.join('..', '..', '..'))
    petsc_arch = _configure_minimal_petsc(petsc_dir)
    docs_loc = _build_classic_docs_subset(petsc_dir, petsc_arch)
    html_extra_dir = _populate_html_extra_from_classic_docs(docs_loc)
    return html_extra_dir


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise


def _configure_minimal_petsc(petsc_dir, petsc_arch='arch-classic-docs') -> None:
    configure = [
        './configure',
        '--with-mpi=0',
        '--with-blaslapack=0',
        '--with-fortran=0',
        '--with-cxx=0',
        '--with-x=0',
        '--with-cmake=0',
        '--with-pthread=0',
        '--with-regexp=0',
        '--download-sowing',
        #'--download-c2html',
        '--with-mkl_sparse_optimize=0',
        '--with-mkl_sparse=0',
        'PETSC_ARCH=' + petsc_arch,
    ]
    if 'READTHEDOCS' not in os.environ:  # Temporary - remove once ReadTheDocs is abandoned and re-add c2html above
        configure.append('--download-c2html')
    print('============================================')
    print('Performing a minimal PETSc (re-)configuration')
    print('PETSC_DIR=%s' % petsc_dir)
    print('PETSC_ARCH=%s' % petsc_arch)
    print('============================================')
    subprocess.run(configure, cwd=petsc_dir, check=True)
    return petsc_arch


def _build_classic_docs_subset(petsc_dir, petsc_arch) -> None:
    docs_loc = os.path.join(os.getcwd(), '_build_classic')
    # Use htmlmap file as a sentinel
    htmlmap_filename = os.path.join(docs_loc, 'docs', 'manualpages', 'htmlmap')
    if os.path.isfile(htmlmap_filename):
        print('============================================')
        print('Assuming that the classic docs in %s are current' % docs_loc)
        print('To rebuild, manually run\n  rm -rf %s' %docs_loc)
        print('============================================')
    else:
        if 'READTHEDOCS' in os.environ:  # Temprary - remove once ReadTheDocs is abandoned
            target = 'allcite'
        else:
            target = 'alldoc12'
        command = ['make', 'alldoc12',
                   'PETSC_DIR=%s' % petsc_dir,
                   'PETSC_ARCH=%s' % petsc_arch,
                   'LOC=%s' % docs_loc]
        print('============================================')
        print('Building a subset of PETSc classic docs')
        print('PETSC_DIR=%s' % petsc_dir)
        print('PETSC_ARCH=%s' % petsc_arch)
        print(command)
        print('============================================')
        subprocess.run(command, cwd=petsc_dir, check=True)
    return docs_loc


def _populate_html_extra_from_classic_docs(docs_loc) -> str:
    html_extra_dir = os.path.join('generated', 'html_extra')
    _mkdir_p(html_extra_dir)
    if 'READTHEDOCS' in os.environ:  # Temporary - remove once ReadTheDocs is abandoned
        subdirs = ['docs']
    else:
        subdirs = ['docs', 'include', 'src']
    for subdir in subdirs:
        target = os.path.join(html_extra_dir, subdir)
        if os.path.isdir(target):
            shutil.rmtree(target)
        source = os.path.join(docs_loc, subdir)
        print('============================================')
        print('Copying directory %s from %s to %s' % (subdir, source, target))
        print('============================================')
        shutil.copytree(source, target)
    return html_extra_dir

if __name__ == "__main__":
    main()
