#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description='Decode one unconstrained SIF problem and solve it with the TAO CUTEst driver.')
    parser.add_argument('--petsc-dir', default=os.environ.get('PETSC_DIR'))
    parser.add_argument('--petsc-arch', default=os.environ.get('PETSC_ARCH'))
    parser.add_argument('--sif', required=True, type=Path)
    parser.add_argument('--driver', required=True, type=Path)
    parser.add_argument('options', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.petsc_dir or args.petsc_arch is None:
        parser.error('Set PETSC_DIR and PETSC_ARCH or supply --petsc-dir and --petsc-arch.')
    options = args.options[1:] if args.options[:1] == ['--'] else args.options
    source = args.sif.resolve(strict=True)
    driver = args.driver.resolve(strict=True)
    makefile = Path(__file__).resolve().with_name('makefile')
    with tempfile.TemporaryDirectory(prefix='petsc-cutest-') as directory:
        result = subprocess.run(['make', '-f', str(makefile), 'all',
                                 'PETSC_DIR='+str(Path(args.petsc_dir).resolve()),
                                 'PETSC_ARCH='+args.petsc_arch, 'SIF='+str(source)],
                                cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            print(result.stdout, end='')
            result.check_returncode()
        library, = Path(directory).glob('libcutest_'+source.stem+'.*')
        subprocess.run([str(driver), '-cutest_lib', str(library),
                        '-cutest_data', str(Path(directory)/'OUTSDIF.d'), *options], check=True)


if __name__ == '__main__':
    main()
