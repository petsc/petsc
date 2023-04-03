#!/usr/bin/env python
""" Configure PETSc and build and place the generated manual pages (as .md files) and html source (as .html files)"""

import os
import errno
import subprocess
import shutil
import argparse


rawhtml = ['include', 'src']
CLASSIC_DOCS_ARCH = 'arch-classic-docs'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main(stage,outdir):
    """ Operations to provide data from the 'classic' PETSc docs system. """
    petsc_dir = os.path.abspath(os.path.join(THIS_DIR, ".."))  # abspath essential since classic 'html' target uses sed to modify paths from the source to target tree
    if stage == "pre":
        _configure_minimal_petsc(petsc_dir, CLASSIC_DOCS_ARCH)
    else:
        if not os.path.isfile(os.path.join(petsc_dir, "configure.log")):
            raise Exception("Expected PETSc configuration not found")
    _build_classic_docs_subset(petsc_dir, CLASSIC_DOCS_ARCH, outdir,stage)

def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise


def _configure_minimal_petsc(petsc_dir, petsc_arch) -> None:
    if 'PETSC_ARCH' in os.environ: del os.environ['PETSC_ARCH']
    if 'MAKEFLAGS' in os.environ: del os.environ['MAKEFLAGS']
    configure = [
        './configure',
        '--with-mpi=0',
        '--with-cxx=0',
        '--with-x=0',
        '--with-cmake=0',
        '--with-pthread=0',
        '--with-regexp=0',
        '--download-sowing',
        '--download-c2html',
        '--with-mkl_sparse_optimize=0',
        '--with-mkl_sparse=0',
        '--with-petsc4py',
        'PETSC_ARCH=' + petsc_arch,
    ]
    print('==================================================================')
    print('Performing a minimal PETSc (re-)configuration needed to build docs')
    print('PETSC_DIR=%s' % petsc_dir)
    print('PETSC_ARCH=%s' % petsc_arch)
    print('==================================================================')
    subprocess.run(configure, cwd=petsc_dir, check=True)
    return petsc_arch


def _build_classic_docs_subset(petsc_dir, petsc_arch, outdir, stage):
    if stage == "pre":
        target = "alldoc_pre"
        loc = os.getcwd()
    elif stage == "post":
        target = "alldoc_post"
        loc = outdir
    else:
        raise Exception("Unrecognized stage %s" % stage)
    command = ['make', target,
               'PETSC_DIR=%s' % petsc_dir,
               'PETSC_ARCH=%s' % petsc_arch,
               'HTMLMAP=%s' % os.path.join(os.getcwd(),'manualpages','htmlmap'),
               'LOC=%s' % loc]
    print('============================================')
    print('Building a subset of PETSc classic docs (%s)' % stage)
    print('PETSC_DIR=%s' % petsc_dir)
    print('PETSC_ARCH=%s' % petsc_arch)
    print(command)
    print('============================================')
    subprocess.run(command, cwd=petsc_dir, check=True)
    if stage ==  "post":
        try:
            subprocess.run(['python2', '--version'])
        except:
            print('Skipping petsc4py documentation build since python2 is not available in PATH')
            return
        rawhtml.append('petsc4py')
        command = ['make', 'all',
                   'PETSC_DIR=%s' % petsc_dir,
                   'PETSC_ARCH=%s' % petsc_arch]
        print('==============================================')
        print('Building library to make petsc4py classic docs')
        print(command)
        print('==============================================')
        subprocess.run(command, cwd=petsc_dir, check=True)
        command = ['make', 'website',
                   'PETSC_DIR=%s' % petsc_dir,
                   'PETSC_ARCH=%s' % petsc_arch,
                   'LOC=%s' % loc]
        print('============================================')
        print('Building petsc4py classic docs')
        print(command)
        print('============================================')
        subprocess.run(command, cwd=os.path.join(petsc_dir,'src','binding','petsc4py'), check=True)


def classic_docs_subdirs(stage):
    if stage == 'pre':   # generated .md files that Sphinx will use to create website
        return ['manualpages']
    if stage == 'post':  # raw .html files that get copied directly to the website (not processed by Sphinx)
        return rawhtml
    raise Exception('Unrecognized stage %s' % stage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', '-c', action='store_true')
    parser.add_argument('--stage', '-s')
    args = parser.parse_args()

    if args.clean:
        clean()
    else:
        main(args.stage)
