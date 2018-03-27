#!/usr/bin/env python

import os
from distutils.sysconfig import parse_makefile
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from cmakegen import Mistakes, stripsplit, AUTODIRS, SKIPDIRS
from cmakegen import defaultdict # collections.defaultdict, with fallback for python-2.4

PKGS = 'sys vec mat dm ksp snes ts tao'.split()
LANGS = dict(c='C', cxx='CXX', cu='CU', F='F', F90='F90')

try:
    all([True, True])
except NameError:               # needs python-2.5
    def all(iterable):
        for i in iterable:
            if not i:
                return False
        return True

try:
    os.path.relpath             # needs python-2.6
except AttributeError:
    def _relpath(path, start=os.path.curdir):
        """Return a relative version of a path"""

        from os.path import curdir, abspath, commonprefix, sep, pardir, join
        if not path:
            raise ValueError("no path specified")

        start_list = [x for x in abspath(start).split(sep) if x]
        path_list = [x for x in abspath(path).split(sep) if x]

        # Work out how much of the filepath is shared by start and path.
        i = len(commonprefix([start_list, path_list]))

        rel_list = [pardir] * (len(start_list)-i) + path_list[i:]
        if not rel_list:
            return curdir
        return join(*rel_list)
    os.path.relpath = _relpath

class debuglogger(object):
    def __init__(self, log):
        self._log = log

    def write(self, string):
        self._log.debug(string)

class Petsc(object):
    def __init__(self, petsc_dir=None, petsc_arch=None, verbose=False):
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
        self.petsc_dir = petsc_dir
        self.petsc_arch = petsc_arch
        self.read_conf()
        try:
            logging.basicConfig(filename=self.arch_path('lib','petsc','conf', 'gmake.log'), level=logging.DEBUG)
        except IOError:
            # Disable logging if path is not writeable (e.g., prefix install)
            logging.basicConfig(filename='/dev/null', level=logging.DEBUG)
        self.log = logging.getLogger('gmakegen')
        self.mistakes = Mistakes(debuglogger(self.log), verbose=verbose)
        self.gendeps = []

    def arch_path(self, *args):
        return os.path.join(self.petsc_dir, self.petsc_arch, *args)

    def read_conf(self):
        self.conf = dict()
        for line in open(self.arch_path('include', 'petscconf.h')):
            if line.startswith('#define '):
                define = line[len('#define '):]
                space = define.find(' ')
                key = define[:space]
                val = define[space+1:]
                self.conf[key] = val
        self.conf.update(parse_makefile(self.arch_path('lib','petsc','conf', 'petscvariables')))
        self.have_fortran = int(self.conf.get('PETSC_HAVE_FORTRAN', '0'))

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
        return os.path.relpath(os.path.join(root, src), self.petsc_dir)

    def get_sources(self, makevars):
        """Return dict {lang: list_of_source_files}"""
        source = dict()
        for lang, sourcelang in LANGS.items():
            source[lang] = [f for f in makevars.get('SOURCE'+sourcelang,'').split() if f.endswith(lang)]
        return source

    def gen_pkg(self, pkg):
        pkgsrcs = dict()
        for lang in LANGS:
            pkgsrcs[lang] = []
        for root, dirs, files in os.walk(os.path.join(self.petsc_dir, 'src', pkg)):
            dirs.sort()
            files.sort()
            makefile = os.path.join(root,'makefile')
            if not os.path.exists(makefile):
                dirs[:] = []
                continue
            mklines = open(makefile)
            conditions = set(tuple(stripsplit(line)) for line in mklines if line.startswith('#requires'))
            mklines.close()
            if not all(self.inconf(key, val) for key, val in conditions):
                dirs[:] = []
                continue
            makevars = parse_makefile(makefile)
            mdirs = makevars.get('DIRS','').split() # Directories specified in the makefile
            self.mistakes.compareDirLists(root, mdirs, dirs) # diagnostic output to find unused directories
            candidates = set(mdirs).union(AUTODIRS).difference(SKIPDIRS)
            dirs[:] = list(candidates.intersection(dirs))
            allsource = []
            def mkrel(src):
                return self.relpath(root, src)
            source = self.get_sources(makevars)
            for lang, s in source.items():
                pkgsrcs[lang] += map(mkrel, s)
                allsource += s
            self.mistakes.compareSourceLists(root, allsource, files) # Diagnostic output about unused source files
            self.gendeps.append(self.relpath(root, 'makefile'))
        return pkgsrcs

    def gen_gnumake(self, fd):
        def write(stem, srcs):
            for lang in LANGS:
                fd.write('%(stem)s.%(lang)s := %(srcs)s\n' % dict(stem=stem, lang=lang, srcs=' '.join(srcs[lang])))
        for pkg in PKGS:
            srcs = self.gen_pkg(pkg)
            write('srcs-' + pkg, srcs)
        return self.gendeps

    def gen_ninja(self, fd):
        libobjs = []
        for pkg in PKGS:
            srcs = self.gen_pkg(pkg)
            for lang in LANGS:
                for src in srcs[lang]:
                    obj = '$objdir/%s.o' % src
                    fd.write('build %(obj)s : %(lang)s_COMPILE %(src)s\n' % dict(obj=obj, lang=lang.upper(), src=os.path.join(self.petsc_dir,src)))
                    libobjs.append(obj)
        fd.write('\n')
        fd.write('build $libdir/libpetsc.so : %s_LINK_SHARED %s\n\n' % ('CF'[self.have_fortran], ' '.join(libobjs)))
        fd.write('build petsc : phony || $libdir/libpetsc.so\n\n')

    def summary(self):
        self.mistakes.summary()

def WriteGnuMake(petsc):
    arch_files = petsc.arch_path('lib','petsc','conf', 'files')
    fd = open(arch_files, 'w')
    gendeps = petsc.gen_gnumake(fd)
    fd.write('\n')
    fd.write('# Dependency to regenerate this file\n')
    fd.write('%s : %s %s\n' % (os.path.relpath(arch_files, petsc.petsc_dir),
                               os.path.relpath(__file__, os.path.realpath(petsc.petsc_dir)),
                               ' '.join(gendeps)))
    fd.write('\n')
    fd.write('# Dummy dependencies in case makefiles are removed\n')
    fd.write(''.join([dep + ':\n' for dep in gendeps]))
    fd.close()

def WriteNinja(petsc):
    conf = dict()
    parse_makefile(os.path.join(petsc.petsc_dir, 'lib', 'petsc','conf', 'variables'), conf)
    parse_makefile(petsc.arch_path('lib','petsc','conf', 'petscvariables'), conf)
    build_ninja = petsc.arch_path('build.ninja')
    fd = open(build_ninja, 'w')
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
                                                       ' '.join(os.path.join(petsc.petsc_dir, dep) for dep in petsc.gendeps)))

def main(petsc_dir=None, petsc_arch=None, output=None, verbose=False):
    if output is None:
        output = 'gnumake'
    writer = dict(gnumake=WriteGnuMake, ninja=WriteNinja)
    petsc = Petsc(petsc_dir=petsc_dir, petsc_arch=petsc_arch, verbose=verbose)
    writer[output](petsc)
    petsc.summary()

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--verbose', help='Show mismatches between makefiles and the filesystem', action='store_true', default=False)
    parser.add_option('--petsc-arch', help='Set PETSC_ARCH different from environment', default=os.environ.get('PETSC_ARCH'))
    parser.add_option('--output', help='Location to write output file', default=None)
    opts, extra_args = parser.parse_args()
    if extra_args:
        import sys
        sys.stderr.write('Unknown arguments: %s\n' % ' '.join(extra_args))
        exit(1)
    main(petsc_arch=opts.petsc_arch, output=opts.output, verbose=opts.verbose)
