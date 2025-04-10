#!/usr/bin/env python3

import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from collections import defaultdict

SKIPDIRS = set('benchmarks build mex-scripts tests tutorials'.split())       # Skip these during the build

def pathsplit(pkg_dir, path):
    """Recursively split a path, returns a tuple"""
    stem, basename = os.path.split(path)
    if stem == '' or stem == pkg_dir:
        return (basename,)
    if stem == path: # fixed point, likely '/'
        return (None,)
    return pathsplit(pkg_dir, stem) + (basename,)

def getlangext(name):
    """Returns everything after the first . in the filename, including the ."""
    file = os.path.basename(name)
    loc = file.find('.')
    if loc > -1: return file[loc:]
    else: return ''

def getlangsplit(name):
    """Returns everything before the first . in the filename, excluding the ."""
    file = os.path.basename(name)
    loc = file.find('.')
    if loc > -1: return os.path.join(os.path.dirname(name),file[:loc])
    raise RuntimeError("No . in filename")

def stripsplit(line):
  return line[len('#requires'):].replace("'","").split()

def parse_makefile(fn, out=None):
    if out is None:
        out = {}
    with open(fn) as f:
        for l in f:
            if "=" in l:
                a,b = l.split("=", 1)
                out[a.strip()] = b.strip()
    return out

PetscPKGS = 'sys vec mat dm ksp snes ts tao ml'.split()
# the key is actually the language suffix, it won't work for suffixes such as 'kokkos.cxx' so use an _ and replace the _ as needed with .
LANGS = dict(kokkos_cxx='KOKKOS', hip_cpp='HIP', sycl_cxx='SYCL', raja_cxx='RAJA', c='C', cxx='CXX', cpp='CPP', cu='CU', F='F', F90='F90')

class debuglogger(object):
    def __init__(self, log):
        self._log = log

    def write(self, string):
        self._log.debug(string)

class Petsc(object):
    def __init__(self, petsc_dir=None, petsc_arch=None, pkg_dir=None, pkg_name=None, pkg_arch=None, pkg_pkgs=None):
        if petsc_dir is None:
            petsc_dir = os.environ.get('PETSC_DIR')
            if petsc_dir is None:
                try:
                    petsc_dir = parse_makefile(os.path.join('lib','petsc','conf', 'petscvariables')).get('PETSC_DIR')
                finally:
                    if petsc_dir is None:
                        raise RuntimeError('Could not determine PETSC_DIR, please set in environment')
        if petsc_arch is None:
            petsc_arch = os.environ.get('PETSC_ARCH')
            if petsc_arch is None:
                try:
                    petsc_arch = parse_makefile(os.path.join(petsc_dir, 'lib','petsc','conf', 'petscvariables')).get('PETSC_ARCH')
                finally:
                    if petsc_arch is None:
                        raise RuntimeError('Could not determine PETSC_ARCH, please set in environment')
        self.petsc_dir = os.path.normpath(petsc_dir)
        self.petsc_arch = petsc_arch.rstrip(os.sep)
        self.pkg_dir = pkg_dir
        self.pkg_name = pkg_name
        self.pkg_arch = pkg_arch
        if self.pkg_dir is None:
          self.pkg_dir = petsc_dir
          self.pkg_name = 'petsc'
          self.pkg_arch = self.petsc_arch
        if self.pkg_name is None:
          self.pkg_name = os.path.basename(os.path.normpath(self.pkg_dir))
        if self.pkg_arch is None:
          self.pkg_arch = self.petsc_arch
        self.pkg_pkgs = PetscPKGS
        if pkg_pkgs is not None:
          if pkg_pkgs.find(',') > 0: npkgs = set(pkg_pkgs.split(','))
          else: npkgs = set(pkg_pkgs.split(' '))
          self.pkg_pkgs += list(npkgs - set(self.pkg_pkgs))
        self.read_conf()
        try:
            logging.basicConfig(filename=self.pkg_arch_path('lib',self.pkg_name,'conf', 'gmake.log'), level=logging.DEBUG)
        except IOError:
            # Disable logging if path is not writeable (e.g., prefix install)
            logging.basicConfig(filename='/dev/null', level=logging.DEBUG)
        self.log = logging.getLogger('gmakegen')
        self.gendeps = []

    def arch_path(self, *args):
        return os.path.join(self.petsc_dir, self.petsc_arch, *args)

    def pkg_arch_path(self, *args):
        return os.path.join(self.pkg_dir, self.pkg_arch, *args)

    def read_conf(self):
        self.conf = dict()
        with open(self.arch_path('include', 'petscconf.h')) as petscconf_h:
            for line in petscconf_h:
                if line.startswith('#define '):
                    define = line[len('#define '):]
                    space = define.find(' ')
                    key = define[:space]
                    val = define[space+1:]
                    self.conf[key] = val
        self.conf.update(parse_makefile(self.arch_path('lib','petsc','conf', 'petscvariables')))
        # allow parsing package additional configurations (if any)
        if self.pkg_name != 'petsc' :
            f = self.pkg_arch_path('include', self.pkg_name + 'conf.h')
            if os.path.isfile(f):
                with open(self.pkg_arch_path('include', self.pkg_name + 'conf.h')) as pkg_conf_h:
                    for line in pkg_conf_h:
                        if line.startswith('#define '):
                            define = line[len('#define '):]
                            space = define.find(' ')
                            key = define[:space]
                            val = define[space+1:]
                            self.conf[key] = val
            f = self.pkg_arch_path('lib',self.pkg_name,'conf', self.pkg_name + 'variables')
            if os.path.isfile(f):
                self.conf.update(parse_makefile(self.pkg_arch_path('lib',self.pkg_name,'conf', self.pkg_name + 'variables')))
        self.have_fortran = int(self.conf.get('PETSC_USE_FORTRAN_BINDINGS', '0'))

    def inconf(self, key, val):
        if key in ['package', 'function', 'define']:
            return self.conf.get(val)
        elif key == 'precision':
            return val == self.conf['PETSC_PRECISION']
        elif key == 'scalar':
            return val == self.conf['PETSC_SCALAR']
        elif key == 'language':
            return val == self.conf['PETSC_LANGUAGE']
        raise RuntimeError('Unknown conf check: %s %s' % (key, val))

    def relpath(self, root, src):
        return os.path.relpath(os.path.join(root, src), self.pkg_dir)

    def get_sources_from_files(self, files):
        """Return dict {lang: list_of_source_files}"""
        source = dict()
        for lang, sourcelang in LANGS.items():
            source[lang] = [f for f in files if f.endswith('.'+lang.replace('_','.'))]
            files = [f for f in files if not f.endswith('.'+lang.replace('_','.'))]
        return source

    def gen_pkg(self, pkg_arch, pkg):
        from itertools import chain
        pkgsrcs = dict()
        for lang in LANGS:
            pkgsrcs[lang] = []
        for root, dirs, files in chain.from_iterable(os.walk(path) for path in [os.path.join(self.pkg_dir, 'src', pkg),os.path.join(self.pkg_dir, self.pkg_arch, 'ftn', pkg)]):
            if SKIPDIRS.intersection(pathsplit(self.pkg_dir, root)): continue
            if not self.have_fortran and os.path.basename(root).find('ftn-') > -1: continue
            dirs.sort()
            dirs[:] = list(set(dirs).difference(SKIPDIRS))
            files.sort()
            makefile = os.path.join(root,'makefile')
            if os.path.isfile(makefile):
              with open(makefile) as mklines:
                conditions = set(tuple(stripsplit(line)) for line in mklines if line.startswith('#requires'))
              if not all(self.inconf(key, val) for key, val in conditions):
                dirs[:] = []
                continue
            allsource = []
            if root.find('/ftn/') > -1: nroot = ''.join(root.rsplit(pkg_arch + '/', 1))
            else: nroot = root
            def mkrel(src):
                return self.relpath(nroot, src)
            if files:
              source = self.get_sources_from_files(files)
              for lang, s in source.items():
                  pkgsrcs[lang] += [mkrel(t) for t in s]
              if os.path.isfile(makefile): self.gendeps.append(self.relpath(root, 'makefile'))
        return pkgsrcs

    def gen_gnumake(self, pkg_arch,fd):
        def write(stem, srcs):
            for lang in LANGS:
                fd.write('%(stem)s.%(lang)s := %(srcs)s\n' % dict(stem=stem, lang=lang.replace('_','.'), srcs=' '.join(sorted(srcs[lang]))))
        for pkg in self.pkg_pkgs:
            srcs = self.gen_pkg(pkg_arch,pkg)
            write('srcs-' + pkg, srcs)
        return self.gendeps

    def gen_ninja(self, fd):
        libobjs = []
        for pkg in self.pkg_pkgs:
            srcs = self.gen_pkg(pkg)
            for lang in LANGS:
                for src in srcs[lang]:
                    obj = '$objdir/%s.o' % src
                    fd.write('build %(obj)s : %(lang)s_COMPILE %(src)s\n' % dict(obj=obj, lang=lang.upper(), src=os.path.join(self.pkg_dir,src)))
                    libobjs.append(obj)
        fd.write('\n')
        fd.write('build $libdir/libpetsc.so : %s_LINK_SHARED %s\n\n' % ('CF'[self.have_fortran], ' '.join(libobjs)))
        fd.write('build petsc : phony || $libdir/libpetsc.so\n\n')

def WriteGnuMake(pkg_arch,petsc):
    arch_files = petsc.pkg_arch_path('lib',petsc.pkg_name,'conf', 'files')
    with open(arch_files, 'w') as fd:
        gendeps = petsc.gen_gnumake(pkg_arch,fd)
        fd.write('\n')
        fd.write('# Dependency to regenerate this file\n')
        fd.write('%s : %s %s\n' % (os.path.relpath(arch_files, petsc.pkg_dir),
                                   os.path.relpath(__file__, os.path.realpath(petsc.pkg_dir)),
                                   ' '.join(gendeps)))
        fd.write('\n')
        fd.write('# Dummy dependencies in case makefiles are removed\n')
        fd.write(''.join([dep + ':\n' for dep in gendeps]))

def WriteNinja(petsc):
    conf = dict()
    parse_makefile(os.path.join(petsc.petsc_dir, 'lib', 'petsc','conf', 'variables'), conf)
    parse_makefile(petsc.arch_path('lib','petsc','conf', 'petscvariables'), conf)
    build_ninja = petsc.arch_path('build.ninja')
    with open(build_ninja, 'w') as fd:
        fd.write('objdir = obj-ninja\n')
        fd.write('libdir = lib\n')
        fd.write('c_compile = %(PCC)s\n' % conf)
        fd.write('c_flags = %(PETSC_CC_INCLUDES)s %(PCC_FLAGS)s %(CCPPFLAGS)s\n' % conf)
        fd.write('c_link = %(PCC_LINKER)s\n' % conf)
        fd.write('c_link_flags = %(PCC_LINKER_FLAGS)s\n' % conf)
        if petsc.have_fortran:
            fd.write('f_compile = %(FC)s\n' % conf)
            fd.write('f_flags = %(PETSC_FC_INCLUDES)s %(FC_FLAGS)s %(FCPPFLAGS)s\n' % conf)
            fd.write('f_link = %(FC_LINKER)s\n' % conf)
            fd.write('f_link_flags = %(FC_LINKER_FLAGS)s\n' % conf)
        fd.write('petsc_external_lib = %(PETSC_EXTERNAL_LIB_BASIC)s\n' % conf)
        fd.write('python = %(PYTHON)s\n' % conf)
        fd.write('\n')
        fd.write('rule C_COMPILE\n'
                 '  command = $c_compile -MMD -MF $out.d $c_flags -c $in -o $out\n'
                 '  description = CC $out\n'
                 '  depfile = $out.d\n'
                 # '  deps = gcc\n') # 'gcc' is default, 'msvc' only recognized by newer versions of ninja
                 '\n')
        fd.write('rule C_LINK_SHARED\n'
                 '  command = $c_link $c_link_flags -shared -o $out $in $petsc_external_lib\n'
                 '  description = CLINK_SHARED $out\n'
                 '\n')
        if petsc.have_fortran:
            fd.write('rule F_COMPILE\n'
                     '  command = $f_compile -MMD -MF $out.d $f_flags -c $in -o $out\n'
                     '  description = FC $out\n'
                     '  depfile = $out.d\n'
                     '\n')
            fd.write('rule F_LINK_SHARED\n'
                     '  command = $f_link $f_link_flags -shared -o $out $in $petsc_external_lib\n'
                     '  description = FLINK_SHARED $out\n'
                     '\n')
        fd.write('rule GEN_NINJA\n'
                 '  command = $python $in --output=ninja\n'
                 '  generator = 1\n'
                 '\n')
        petsc.gen_ninja(fd)
        fd.write('\n')
        fd.write('build %s : GEN_NINJA | %s %s %s %s\n' % (build_ninja,
                                                           os.path.abspath(__file__),
                                                           os.path.join(petsc.petsc_dir, 'lib','petsc','conf', 'variables'),
                                                           petsc.arch_path('lib','petsc','conf', 'petscvariables'),
                                                       ' '.join(os.path.join(petsc.pkg_dir, dep) for dep in petsc.gendeps)))

def main(petsc_dir=None, petsc_arch=None, pkg_dir=None, pkg_name=None, pkg_arch=None, pkg_pkgs=None, output=None):
    petsc = Petsc(petsc_dir=petsc_dir, petsc_arch=petsc_arch, pkg_dir=pkg_dir, pkg_name=pkg_name, pkg_arch=pkg_arch, pkg_pkgs=pkg_pkgs)
    # Use pkg_arch in case petsc_arch is empty (needed by SLEPc)
    WriteGnuMake(petsc_arch if petsc_arch else pkg_arch,petsc)

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--petsc-arch', help='Set PETSC_ARCH different from environment', default=os.environ.get('PETSC_ARCH'))
    parser.add_option('--pkg-dir', help='Set the directory of the package (different from PETSc) you want to generate the makefile rules for', default=None)
    parser.add_option('--pkg-name', help='Set the name of the package you want to generate the makefile rules for', default=None)
    parser.add_option('--pkg-arch', help='Set the package arch name you want to generate the makefile rules for', default=None)
    parser.add_option('--pkg-pkgs', help='Set the package folders (comma separated list, different from the usual sys,vec,mat etc) you want to generate the makefile rules for', default=None)
    parser.add_option('--output', help='Location to write output file', default=None)
    opts, extra_args = parser.parse_args()
    if extra_args:
        import sys
        sys.stderr.write('Unknown arguments: %s\n' % ' '.join(extra_args))
        exit(1)
    main(petsc_arch=opts.petsc_arch, pkg_dir=opts.pkg_dir, pkg_name=opts.pkg_name, pkg_arch=opts.pkg_arch, pkg_pkgs=opts.pkg_pkgs, output=opts.output)
