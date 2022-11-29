# --------------------------------------------------------------------

__all__ = ['PetscConfig',
           'setup', 'Extension',
           'config', 'build', 'build_src', 'build_ext',
           'clean', 'test', 'sdist',
           'log',
           ]

# --------------------------------------------------------------------

import sys, os
import re
try:
    import setuptools
except ImportError:
    setuptools = None

def import_command(cmd):
    try:
        from importlib import import_module
    except ImportError:
        import_module = lambda n:  __import__(n, fromlist=[None])
    try:
        if not setuptools: raise ImportError
        mod = import_module('setuptools.command.' + cmd)
        return getattr(mod, cmd)
    except ImportError:
        mod = import_module('distutils.command.' + cmd)
        return getattr(mod, cmd)

if setuptools:
    from setuptools import setup
    from setuptools import Extension as _Extension
    from setuptools import Command
else:
    from distutils.core import setup
    from distutils.core import Extension as _Extension
    from distutils.core import Command

_config    = import_command('config')
_build     = import_command('build')
_build_ext = import_command('build_ext')
_install   = import_command('install')
_clean     = import_command('clean')
_sdist     = import_command('sdist')

from distutils import sysconfig
from distutils import log
from distutils.util import split_quoted, execute
from distutils.errors import DistutilsError

# --------------------------------------------------------------------

def fix_config_vars(names, values):
    import os, re
    values = list(values)
    if 'CONDA_BUILD' in os.environ:
        return values
    if sys.platform == 'darwin':
        if 'ARCHFLAGS' in os.environ:
            ARCHFLAGS = os.environ['ARCHFLAGS']
            for i, flag in enumerate(list(values)):
                flag, count = re.subn('-arch\s+\w+', ' ', str(flag))
                if count and ARCHFLAGS:
                    flag = flag + ' ' + ARCHFLAGS
                values[i] = flag
        if 'SDKROOT' in os.environ:
            SDKROOT = os.environ['SDKROOT']
            for i, flag in enumerate(list(values)):
                flag, count = re.subn('-isysroot [^ \t]*', ' ', str(flag))
                if count and SDKROOT:
                    flag = flag + ' ' + '-isysroot ' + SDKROOT
                values[i] = flag
    return values

def get_config_vars(*names):
    # Core Python configuration
    values = sysconfig.get_config_vars(*names)
    # Do any distutils flags fixup right now
    values = fix_config_vars(names, values)
    return values

from distutils.unixccompiler import UnixCCompiler
rpath_option_orig = UnixCCompiler.runtime_library_dir_option
def rpath_option(compiler, dir):
    option = rpath_option_orig(compiler, dir)
    if sys.platform[:5] == 'linux':
        if option.startswith('-R'):
            option =  option.replace('-R', '-Wl,-rpath,', 1)
        elif option.startswith('-Wl,-R'):
            option =  option.replace('-Wl,-R', '-Wl,-rpath,', 1)
    return option
UnixCCompiler.runtime_library_dir_option = rpath_option

# --------------------------------------------------------------------

class PetscConfig:

    def __init__(self, petsc_dir, petsc_arch, dest_dir=None):
        if dest_dir is None:
            dest_dir = os.environ.get('DESTDIR')
        self.configdict = { }
        if not petsc_dir:
            raise DistutilsError("PETSc not found")
        if not os.path.isdir(petsc_dir):
            raise DistutilsError("invalid PETSC_DIR: %s" % petsc_dir)
        self.version    = self._get_petsc_version(petsc_dir)
        self.configdict = self._get_petsc_config(petsc_dir, petsc_arch)
        self.PETSC_DIR  = self['PETSC_DIR']
        self.PETSC_ARCH = self['PETSC_ARCH']
        self.DESTDIR = dest_dir
        language_map = {'CONLY':'c', 'CXXONLY':'c++'}
        self.language = language_map[self['PETSC_LANGUAGE']]

    def __getitem__(self, item):
        return self.configdict[item]

    def get(self, item, default=None):
        return self.configdict.get(item, default)

    def configure(self, extension, compiler=None):
        self.configure_extension(extension)
        if compiler is not None:
            self.configure_compiler(compiler)

    def _get_petsc_version(self, petsc_dir):
        import re
        version_re = {
            'major'  : re.compile(r"#define\s+PETSC_VERSION_MAJOR\s+(\d+)"),
            'minor'  : re.compile(r"#define\s+PETSC_VERSION_MINOR\s+(\d+)"),
            'micro'  : re.compile(r"#define\s+PETSC_VERSION_SUBMINOR\s+(\d+)"),
            'release': re.compile(r"#define\s+PETSC_VERSION_RELEASE\s+(-*\d+)"),
            }
        petscversion_h = os.path.join(petsc_dir, 'include', 'petscversion.h')
        with open(petscversion_h, 'rt') as f: data = f.read()
        major = int(version_re['major'].search(data).groups()[0])
        minor = int(version_re['minor'].search(data).groups()[0])
        micro = int(version_re['micro'].search(data).groups()[0])
        release = int(version_re['release'].search(data).groups()[0])
        return  (major, minor, micro), (release == 1)

    def _get_petsc_config(self, petsc_dir, petsc_arch):
        from os.path import join, isdir, exists
        PETSC_DIR  = petsc_dir
        PETSC_ARCH = petsc_arch
        #
        confdir = join('lib', 'petsc', 'conf')
        if not (PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH))):
            petscvars = join(PETSC_DIR, confdir, 'petscvariables')
            PETSC_ARCH = makefile(open(petscvars, 'rt')).get('PETSC_ARCH')
        if not (PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH))):
            PETSC_ARCH = ''
        #
        variables = join(PETSC_DIR, confdir, 'variables')
        if not exists(variables):
            variables  = join(PETSC_DIR, PETSC_ARCH, confdir, 'variables')
        petscvariables = join(PETSC_DIR, PETSC_ARCH, confdir, 'petscvariables')
        #
        with open(variables) as f:
            contents = f.read()
        with open(petscvariables) as f:
            contents += f.read()
        #
        try:
            from cStringIO import StringIO
        except ImportError:
            from io import StringIO
        confstr  = 'PETSC_DIR  = %s\n' % PETSC_DIR
        confstr += 'PETSC_ARCH = %s\n' % PETSC_ARCH
        confstr += contents
        confdict = makefile(StringIO(confstr))
        return confdict

    def _configure_ext(self, ext, dct, preppend=False):
        extdict = ext.__dict__
        for key, values in dct.items():
            if key in extdict:
                for value in values:
                    if value not in extdict[key]:
                        if preppend:
                            extdict[key].insert(0, value)
                        else:
                            extdict[key].append(value)

    def configure_extension(self, extension):
        # includes and libraries
        # paths in PETSc config files point to final installation location, but
        # we might be building against PETSc in staging location (DESTDIR) when
        # DESTDIR is set, so append DESTDIR (if nonempty) to those paths
        petsc_inc = flaglist(prepend_to_flags(self.DESTDIR, self['PETSC_CC_INCLUDES']))
        lib_flags = prepend_to_flags(self.DESTDIR, '-L%s %s' % \
                (self['PETSC_LIB_DIR'], self['PETSC_LIB_BASIC']))
        petsc_lib = flaglist(lib_flags)
        # runtime_library_dirs is not supported on Windows
        if sys.platform != 'win32':
            # if DESTDIR is set, then we're building against PETSc in a staging
            # directory, but rpath needs to point to final install directory.
            rpath = strip_prefix(self.DESTDIR, self['PETSC_LIB_DIR'])
            petsc_lib['runtime_library_dirs'].append(rpath)

        # Link in extra libraries on static builds
        if self['BUILDSHAREDLIB'] != 'yes':
            petsc_ext_lib = split_quoted(self['PETSC_EXTERNAL_LIB_BASIC'])
            petsc_lib['extra_link_args'].extend(petsc_ext_lib)

        self._configure_ext(extension, petsc_inc, preppend=True)
        self._configure_ext(extension, petsc_lib)

    def configure_compiler(self, compiler):
        if compiler.compiler_type != 'unix': return
        getenv = os.environ.get
        # distutils C/C++ compiler
        (cc, cflags, ccshared, cxx) = get_config_vars(
            'CC', 'CFLAGS',  'CCSHARED', 'CXX')
        ccshared = getenv('CCSHARED', ccshared or '')
        cflags = getenv('CFLAGS', cflags or '')
        cflags = cflags.replace('-Wstrict-prototypes', '')
        # distutils linker
        (ldflags, ldshared, so_ext) = get_config_vars(
            'LDFLAGS', 'LDSHARED', 'SO')
        ld = cc
        ldshared = getenv('LDSHARED', ldshared)
        ldflags = getenv('LDFLAGS', cflags + ' ' + (ldflags or ''))
        ldcmd = split_quoted(ld) + split_quoted(ldflags)
        ldshared = [flg for flg in split_quoted(ldshared) if flg not in ldcmd]
        ldshared = str.join(' ', ldshared)
        #
        def get_flags(cmd):
            if not cmd: return ''
            cmd = split_quoted(cmd)
            if os.path.basename(cmd[0]) == 'xcrun':
                del cmd[0]
                while True:
                    if cmd[0] == '-sdk':
                        del cmd[0:2]
                        continue
                    if cmd[0] == '-log':
                        del cmd[0]
                        continue
                    break
            return ' '.join(cmd[1:])
        # PETSc C compiler
        PCC = self['PCC']
        PCC_FLAGS = get_flags(cc) + ' ' + self['PCC_FLAGS']
        PCC_FLAGS = PCC_FLAGS.replace('-fvisibility=hidden', '')
        PCC = getenv('PCC', PCC) + ' ' +  getenv('PCCFLAGS', PCC_FLAGS)
        PCC_SHARED = str.join(' ', (PCC, ccshared, cflags))
        # PETSc C++ compiler
        PCXX = PCC if self.language == 'c++' else self.get('CXX', cxx)
        # PETSc linker
        PLD = self['PCC_LINKER']
        PLD_FLAGS = get_flags(ld) + ' ' + self['PCC_LINKER_FLAGS']
        PLD_FLAGS = PLD_FLAGS.replace('-fvisibility=hidden', '')
        PLD = getenv('PLD', PLD) + ' ' + getenv('PLDFLAGS', PLD_FLAGS)
        PLD_SHARED = str.join(' ', (PLD, ldshared, ldflags))
        #
        compiler.set_executables(
            compiler     = PCC,
            compiler_cxx = PCXX,
            linker_exe   = PLD,
            compiler_so  = PCC_SHARED,
            linker_so    = PLD_SHARED,
            )
        compiler.shared_lib_extension = so_ext
        #
        if sys.platform == 'darwin':
            for attr in ('preprocessor',
                         'compiler', 'compiler_cxx', 'compiler_so',
                         'linker_so', 'linker_exe'):
                compiler_cmd = getattr(compiler, attr, [])
                while '-mno-fused-madd' in compiler_cmd:
                    compiler_cmd.remove('-mno-fused-madd')

    def log_info(self):
        PETSC_DIR  = self['PETSC_DIR']
        PETSC_ARCH = self['PETSC_ARCH']
        version = ".".join([str(i) for i in self.version[0]])
        release = ("development", "release")[self.version[1]]
        version_info = version + ' ' + release
        integer_size = '%s-bit' % self['PETSC_INDEX_SIZE']
        scalar_type  = self['PETSC_SCALAR']
        precision    = self['PETSC_PRECISION']
        language     = self['PETSC_LANGUAGE']
        compiler     = self['PCC']
        linker       = self['PCC_LINKER']
        log.info('PETSC_DIR:    %s' % PETSC_DIR )
        log.info('PETSC_ARCH:   %s' % PETSC_ARCH )
        log.info('version:      %s' % version_info)
        log.info('integer-size: %s' % integer_size)
        log.info('scalar-type:  %s' % scalar_type)
        log.info('precision:    %s' % precision)
        log.info('language:     %s' % language)
        log.info('compiler:     %s' % compiler)
        log.info('linker:       %s' % linker)

# --------------------------------------------------------------------

class Extension(_Extension):
    pass

# --------------------------------------------------------------------

cmd_petsc_opts = [
    ('petsc-dir=', None,
     "define PETSC_DIR, overriding environmental variables"),
    ('petsc-arch=', None,
     "define PETSC_ARCH, overriding environmental variables"),
    ]

class config(_config):

    Configure = PetscConfig

    user_options = _config.user_options + cmd_petsc_opts

    def initialize_options(self):
        _config.initialize_options(self)
        self.petsc_dir  = None
        self.petsc_arch = None

    def get_config_arch(self, arch):
        return config.Configure(self.petsc_dir, arch)

    def run(self):
        _config.run(self)
        self.petsc_dir = config.get_petsc_dir(self.petsc_dir)
        if self.petsc_dir is None: return
        petsc_arch = config.get_petsc_arch(self.petsc_dir, self.petsc_arch)
        log.info('-' * 70)
        log.info('PETSC_DIR:   %s' % self.petsc_dir)
        arch_list = petsc_arch
        if not arch_list :
            arch_list = [ None ]
        for arch in arch_list:
            conf = self.get_config_arch(arch)
            archname    = conf.PETSC_ARCH or conf['PETSC_ARCH']
            scalar_type = conf['PETSC_SCALAR']
            precision   = conf['PETSC_PRECISION']
            language    = conf['PETSC_LANGUAGE']
            compiler    = conf['PCC']
            linker      = conf['PCC_LINKER']
            log.info('-'*70)
            log.info('PETSC_ARCH:  %s' % archname)
            log.info(' * scalar-type: %s' % scalar_type)
            log.info(' * precision:   %s' % precision)
            log.info(' * language:    %s' % language)
            log.info(' * compiler:    %s' % compiler)
            log.info(' * linker:      %s' % linker)
        log.info('-' * 70)

    #@staticmethod
    def get_petsc_dir(petsc_dir):
        if not petsc_dir: return None
        petsc_dir = os.path.expandvars(petsc_dir)
        if not petsc_dir or '$PETSC_DIR' in petsc_dir:
            try:
                import petsc
                petsc_dir = petsc.get_petsc_dir()
            except ImportError:
                log.warn("PETSC_DIR not specified")
                return None
        petsc_dir = os.path.expanduser(petsc_dir)
        petsc_dir = os.path.abspath(petsc_dir)
        return config.chk_petsc_dir(petsc_dir)
    get_petsc_dir = staticmethod(get_petsc_dir)

    #@staticmethod
    def chk_petsc_dir(petsc_dir):
        if not os.path.isdir(petsc_dir):
            log.error('invalid PETSC_DIR: %s (ignored)' % petsc_dir)
            return None
        return petsc_dir
    chk_petsc_dir = staticmethod(chk_petsc_dir)

    #@staticmethod
    def get_petsc_arch(petsc_dir, petsc_arch):
        if not petsc_dir: return None
        petsc_arch = os.path.expandvars(petsc_arch)
        if (not petsc_arch or '$PETSC_ARCH' in petsc_arch):
            petsc_arch = ''
            petsc_conf = os.path.join(petsc_dir, 'lib', 'petsc', 'conf')
            if os.path.isdir(petsc_conf):
                petscvariables = os.path.join(petsc_conf, 'petscvariables')
                if os.path.exists(petscvariables):
                    conf = makefile(open(petscvariables, 'rt'))
                    petsc_arch = conf.get('PETSC_ARCH', '')
        petsc_arch = petsc_arch.split(os.pathsep)
        petsc_arch = unique(petsc_arch)
        petsc_arch = [arch for arch in petsc_arch if arch]
        return config.chk_petsc_arch(petsc_dir, petsc_arch)
    get_petsc_arch = staticmethod(get_petsc_arch)

    #@staticmethod
    def chk_petsc_arch(petsc_dir, petsc_arch):
        valid_archs = []
        for arch in petsc_arch:
            arch_path = os.path.join(petsc_dir, arch)
            if os.path.isdir(arch_path):
                valid_archs.append(arch)
            else:
                log.warn("invalid PETSC_ARCH: %s (ignored)" % arch)
        return valid_archs
    chk_petsc_arch = staticmethod(chk_petsc_arch)


class build(_build):

    user_options = _build.user_options + cmd_petsc_opts

    def initialize_options(self):
        _build.initialize_options(self)
        self.petsc_dir  = None
        self.petsc_arch = None

    def finalize_options(self):
        _build.finalize_options(self)
        self.set_undefined_options('config',
                                   ('petsc_dir',  'petsc_dir'),
                                   ('petsc_arch', 'petsc_arch'))
        self.petsc_dir  = config.get_petsc_dir(self.petsc_dir)
        self.petsc_arch = config.get_petsc_arch(self.petsc_dir,
                                                self.petsc_arch)

    sub_commands = \
        [('build_src', lambda *args: True)] + \
        _build.sub_commands

class build_src(Command):
    description = "build C sources from Cython files"
    user_options = [
        ('force', 'f',
         "forcibly build everything (ignore file timestamps)"),
        ]
    boolean_options = ['force']
    def initialize_options(self):
        self.force = False
    def finalize_options(self):
        self.set_undefined_options('build',
                                   ('force', 'force'),
                                   )
    def run(self):
        pass

class build_ext(_build_ext):

    user_options = _build_ext.user_options + cmd_petsc_opts

    def initialize_options(self):
        _build_ext.initialize_options(self)
        self.petsc_dir  = None
        self.petsc_arch = None
        self._outputs = []

    def finalize_options(self):
        _build_ext.finalize_options(self)
        self.set_undefined_options('build',
                                   ('petsc_dir',  'petsc_dir'),
                                   ('petsc_arch', 'petsc_arch'))
        if ((sys.platform.startswith('linux') or
             sys.platform.startswith('gnu') or
             sys.platform.startswith('sunos')) and
            sysconfig.get_config_var('Py_ENABLE_SHARED')):
            py_version = sysconfig.get_python_version()
            bad_pylib_dir = os.path.join(sys.prefix, "lib",
                                         "python" + py_version,
                                         "config")
            try:
                self.library_dirs.remove(bad_pylib_dir)
            except ValueError:
                pass
            pylib_dir = sysconfig.get_config_var("LIBDIR")
            if pylib_dir not in self.library_dirs:
                self.library_dirs.append(pylib_dir)
            if pylib_dir not in self.rpath:
                self.rpath.append(pylib_dir)
            if sys.exec_prefix == '/usr':
                self.library_dirs.remove(pylib_dir)
                self.rpath.remove(pylib_dir)

    def _copy_ext(self, ext):
        from copy import deepcopy
        extclass = ext.__class__
        fullname = self.get_ext_fullname(ext.name)
        modpath = str.split(fullname, '.')
        pkgpath = os.path.join('', *modpath[0:-1])
        name = modpath[-1]
        sources = list(ext.sources)
        newext = extclass(name, sources)
        newext.__dict__.update(deepcopy(ext.__dict__))
        newext.name = name
        return pkgpath, newext

    def _build_ext_arch(self, ext, pkgpath, arch):
        build_temp = self.build_temp
        build_lib  = self.build_lib
        try:
            self.build_temp = os.path.join(build_temp, arch)
            self.build_lib  = os.path.join(build_lib, pkgpath, arch)
            _build_ext.build_extension(self, ext)
        finally:
            self.build_temp = build_temp
            self.build_lib  = build_lib

    def get_config_arch(self, arch):
        return config.Configure(self.petsc_dir, arch)

    def build_extension(self, ext):
        if not isinstance(ext, Extension):
            return _build_ext.build_extension(self, ext)
        petsc_arch = self.petsc_arch
        if not petsc_arch:
            petsc_arch = [ None ]
        for arch in petsc_arch:
            config = self.get_config_arch(arch)
            ARCH = arch or config['PETSC_ARCH']
            if ARCH not in self.PETSC_ARCH_LIST:
                self.PETSC_ARCH_LIST.append(ARCH)
            self.DESTDIR = config.DESTDIR
            ext.language = config.language
            config.log_info()
            pkgpath, newext = self._copy_ext(ext)
            config.configure(newext, self.compiler)
            name =  self.distribution.get_name()
            version = self.distribution.get_version()
            distdir = "%s-%s/" % (name, version)
            self._build_ext_arch(newext, pkgpath, ARCH)

    def build_extensions(self, *args, **kargs):
        self.PETSC_ARCH_LIST = []
        _build_ext.build_extensions(self, *args,**kargs)
        if not self.PETSC_ARCH_LIST: return
        self.build_configuration(self.PETSC_ARCH_LIST)


    def build_configuration(self, arch_list):
        #
        template, variables = self.get_config_data(arch_list)
        config_data = template % variables
        #
        build_lib   = self.build_lib
        dist_name   = self.distribution.get_name()
        config_file = os.path.join(build_lib, dist_name, 'lib',
                                   dist_name.replace('4py', '') + '.cfg')
        #
        def write_file(filename, data):
            with open(filename, 'w') as fh:
                fh.write(config_data)
        execute(write_file, (config_file, config_data),
                msg='writing %s' % config_file,
                verbose=self.verbose, dry_run=self.dry_run)

    def get_config_data(self, arch_list):
        template = """\
PETSC_DIR  = %(PETSC_DIR)s
PETSC_ARCH = %(PETSC_ARCH)s
"""
        variables = {'PETSC_DIR'  : strip_prefix(self.DESTDIR, self.petsc_dir),
                     'PETSC_ARCH' : os.path.pathsep.join(arch_list)}
        return template, variables

    def get_outputs(self):
        self.check_extensions_list(self.extensions)
        outputs = []
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            if isinstance(ext, Extension) and self.petsc_arch:
                head, tail = os.path.split(filename)
                for arch in self.petsc_arch:
                    outfile = os.path.join(self.build_lib,
                                           head, arch, tail)
                    outputs.append(outfile)
            else:
                outfile = os.path.join(self.build_lib, filename)
                outputs.append(outfile)
        outputs = list(set(outputs))
        return outputs

class install(_install):
    def run(self):
        _install.run(self)

class clean(_clean):
    def run(self):
        _clean.run(self)
        from distutils.dir_util import remove_tree
        if self.all:
            # remove the <package>.egg_info directory
            try:
                egg_info = self.get_finalized_command('egg_info').egg_info
                if os.path.exists(egg_info):
                    remove_tree(egg_info, dry_run=self.dry_run)
                else:
                    log.debug("'%s' does not exist -- can't clean it",
                              egg_info)
            except DistutilsError:
                pass

class test(Command):
    description = "run the test suite"
    user_options = [('args=', None, "options")]
    def initialize_options(self):
        self.args = None
    def finalize_options(self):
        if self.args:
            self.args = split_quoted(self.args)
        else:
            self.args = []
    def run(self):
        pass

class sdist(_sdist):
    def run(self):
        build_src = self.get_finalized_command('build_src')
        build_src.run()
        _sdist.run(self)

# --------------------------------------------------------------------

if setuptools:
    try:
        from setuptools.command import egg_info as mod_egg_info
        _FileList = mod_egg_info.FileList
        class FileList(_FileList):
            def process_template_line(self, line):
                level = log.set_threshold(log.ERROR)
                try:
                    _FileList.process_template_line(self, line)
                finally:
                    log.set_threshold(level)
        mod_egg_info.FileList = FileList
    except:
        pass

# --------------------------------------------------------------------

def append(seq, item):
    if item not in seq:
        seq.append(item)

def append_dict(conf, dct):
    for key, values in dct.items():
        if key in conf:
            for value in values:
                if value not in conf[key]:
                    conf[key].append(value)
def unique(seq):
    res = []
    for item in seq:
        if item not in res:
            res.append(item)
    return res

def flaglist(flags):

    conf = {
        'define_macros'       : [],
        'undef_macros'        : [],
        'include_dirs'        : [],

        'libraries'           : [],
        'library_dirs'        : [],
        'runtime_library_dirs': [],

        'extra_compile_args'  : [],
        'extra_link_args'     : [],
        }

    if type(flags) is str:
        flags = flags.split()

    switch = '-Wl,'
    newflags = []
    linkopts = []
    for f in flags:
        if f.startswith(switch):
            if len(f) > 4:
                append(linkopts, f[4:])
        else:
            append(newflags, f)
    if linkopts:
        newflags.append(switch + ','.join(linkopts))
    flags = newflags

    append_next_word = None

    for word in flags:

        if append_next_word is not None:
            append(append_next_word, word)
            append_next_word = None
            continue

        switch, value = word[0:2], word[2:]

        if switch == "-I":
            append(conf['include_dirs'], value)
        elif switch == "-D":
            try:
                idx = value.index("=")
                macro = (value[:idx], value[idx+1:])
            except ValueError:
                macro = (value, None)
            append(conf['define_macros'], macro)
        elif switch == "-U":
            append(conf['undef_macros'], value)
        elif switch == "-l":
            append(conf['libraries'], value)
        elif switch == "-L":
            append(conf['library_dirs'], value)
        elif switch == "-R":
            append(conf['runtime_library_dirs'], value)
        elif word.startswith("-Wl"):
            linkopts = word.split(',')
            append_dict(conf, flaglist(linkopts[1:]))
        elif word == "-rpath":
            append_next_word = conf['runtime_library_dirs']
        elif word == "-Xlinker":
            append_next_word = conf['extra_link_args']
        else:
            #log.warn("unrecognized flag '%s'" % word)
            pass
    return conf

def prepend_to_flags(path, flags):
    """Prepend a path to compiler flags with absolute paths"""
    if not path:
        return flags
    def append_path(m):
        switch = m.group(1)
        open_quote = m.group(4)
        old_path = m.group(5)
        close_quote = m.group(6)
        if os.path.isabs(old_path):
            moded_path = os.path.normpath(path + os.path.sep + old_path)
            return switch + open_quote + moded_path + close_quote
        return m.group(0)
    return re.sub(r'((^|\s+)(-I|-L))(\s*["\']?)(\S+)(["\']?)',
            append_path, flags)

def strip_prefix(prefix, string):
    if not prefix:
        return string
    return re.sub(r'^' + prefix, '', string)

# --------------------------------------------------------------------

from distutils.text_file import TextFile

# Regexes needed for parsing Makefile-like syntaxes
import re as _re
_variable_rx = _re.compile("([a-zA-Z][a-zA-Z0-9_]+)\s*=\s*(.*)")
_findvar1_rx = _re.compile(r"\$\(([A-Za-z][A-Za-z0-9_]*)\)")
_findvar2_rx = _re.compile(r"\${([A-Za-z][A-Za-z0-9_]*)}")

def makefile(fileobj, dct=None):
    """Parse a Makefile-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    fp = TextFile(file=fileobj,
                  strip_comments=1,
                  skip_blanks=1,
                  join_lines=1)

    if dct is None:
        dct = {}
    done = {}
    notdone = {}

    while 1:
        line = fp.readline()
        if line is None: # eof
            break
        m = _variable_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            v = str.strip(v)
            if "$" in v:
                notdone[n] = v
            else:
                try: v = int(v)
                except ValueError: pass
                done[n] = v
                try: del notdone[n]
                except KeyError: pass
    fp.close()

    # do variable interpolation here
    while notdone:
        for name in list(notdone.keys()):
            value = notdone[name]
            m = _findvar1_rx.search(value) or _findvar2_rx.search(value)
            if m:
                n = m.group(1)
                found = True
                if n in done:
                    item = str(done[n])
                elif n in notdone:
                    # get it on a subsequent round
                    found = False
                else:
                    done[n] = item = ""
                if found:
                    after = value[m.end():]
                    value = value[:m.start()] + item + after
                    if "$" in after:
                        notdone[name] = value
                    else:
                        try: value = int(value)
                        except ValueError:
                            done[name] = str.strip(value)
                        else:
                            done[name] = value
                        del notdone[name]
            else:
                # bogus variable reference;
                # just drop it since we can't deal
                del notdone[name]
    # save the results in the global dictionary
    dct.update(done)
    return dct

# --------------------------------------------------------------------
