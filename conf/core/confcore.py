# --------------------------------------------------------------------

__all__ = ['PetscConfig',
           'setup', 'Extension',
           'log', 'config',
           'build', 'build_ext',
           ]

# --------------------------------------------------------------------

import sys, os
from cStringIO import StringIO

from distutils.core import setup
from distutils.core import Extension as _Extension
from distutils.command.config    import config     as _config
from distutils.command.build     import build      as _build
from distutils.command.build_ext import build_ext  as _build_ext
from distutils.util import split_quoted
from distutils import log
from distutils.errors import DistutilsError

import confutils as cfgutils

# --------------------------------------------------------------------

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

    def __init__(self, petsc_dir, petsc_arch):
        self.configdict = { }
        if not petsc_dir:
            raise DistutilsError("PETSc not found")
        elif not os.path.isdir(petsc_dir):
            raise DistutilsError("invalid PETSC_DIR: %s" % petsc_dir)
        self.configdict = self._get_petsc_conf(petsc_dir, petsc_arch)
        self.PETSC_DIR  = self['PETSC_DIR']
        self.PETSC_ARCH = self['PETSC_ARCH']
        language_map = {'CONLY':'c', 'CXXONLY':'c++'}
        self.language = language_map[self['PETSC_LANGUAGE']]

    def __getitem__(self, item):
        return self.configdict[item]

    def configure(self, extension, compiler=None):
        self.configure_extension(extension)
        if compiler is not None:
            self.configure_compiler(compiler)

    def _get_petsc_conf(self, petsc_dir, petsc_arch):
        bmake_dir = os.path.join(petsc_dir, 'bmake')
        if os.path.isdir(bmake_dir):
            return self._get_petsc_conf_old(petsc_dir, petsc_arch)
        else:
            return self._get_petsc_conf_new(petsc_dir, petsc_arch)

    def _get_petsc_conf_old(self, petsc_dir, petsc_arch):
        variables = os.path.join(petsc_dir, 'bmake','common',    'variables')
        petscconf = os.path.join(petsc_dir, 'bmake', petsc_arch, 'petscconf')
        #
        variables = open(variables)
        contents = variables.read()
        variables.close()
        petscconf = open(petscconf)
        contents += petscconf.read()
        petscconf.close()
        #
        confstr  = 'PETSC_DIR = %s\n'  % petsc_dir
        confstr += 'PETSC_ARCH = %s\n' % petsc_arch
        confstr += contents
        confdct = cfgutils.makefile(StringIO(confstr))
        return confdct

    def _get_petsc_conf_new(self, petsc_dir, petsc_arch):
        PETSC_DIR = petsc_dir
        if (not petsc_arch or
            not os.path.isdir(os.path.join(petsc_dir, petsc_arch))):
            PETSC_ARCH    = ''
            PETSC_INCLUDE = ['-I${PETSC_DIR}/include']
            PETSC_LIB_DIR = ['${PETSC_DIR}/lib']
        else:
            PETSC_ARCH    = petsc_arch
            PETSC_INCLUDE = ['-I${PETSC_DIR}/include',
                             '-I${PETSC_DIR}/${PETSC_ARCH}/include']
            PETSC_LIB_DIR = ['${PETSC_DIR}/${PETSC_ARCH}/lib']
        PETSC_INCLUDE += ['${PACKAGES_INCLUDES}',
                          '${PETSC_BLASLAPACK_FLAGS}']
        #
        variables = os.path.join(PETSC_DIR,
                                 'conf', 'variables')
        if not os.path.exists(variables):
            variables = os.path.join(PETSC_DIR, PETSC_ARCH,
                                     'conf', 'variables')
        petscvars = os.path.join(PETSC_DIR, PETSC_ARCH,
                                 'conf', 'petscvariables')
        #
        variables = open(variables)
        try: contents = variables.read()
        finally: variables.close()
        petscvars = open(petscvars)
        try: contents += petscvars.read()
        finally: petscvars.close()
        #
        confstr  = 'PETSC_DIR  = %s\n' % PETSC_DIR
        confstr += 'PETSC_ARCH = %s\n' % PETSC_ARCH
        confstr += contents
        confstr += 'PETSC_INCLUDE = %s\n' % ' '.join(PETSC_INCLUDE)
        confstr += 'PETSC_LIB_DIR = %s\n' % ' '.join(PETSC_LIB_DIR)
        confdict = cfgutils.makefile(StringIO(confstr))
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
        # define macros
        macros = [('PETSC_DIR',  self['PETSC_DIR'])]
        extension.define_macros.extend(macros)
        # includes and libraries
        petsc_inc = cfgutils.flaglist(self['PETSC_INCLUDE'])
        petsc_lib = cfgutils.flaglist(
            '-L%s %s' % (self['PETSC_LIB_DIR'], self['PETSC_LIB_BASIC']))
        petsc_lib['runtime_library_dirs'].append(self['PETSC_LIB_DIR'])
        self._configure_ext(extension, petsc_inc, preppend=True)
        self._configure_ext(extension, petsc_lib)
        # extra compiler and linker configuration
        ccflags = split_quoted(self['PCC_FLAGS'])
        if sys.version_info[:2] < (2, 5):
            try: ccflags.remove('-Wwrite-strings')
            except ValueError: pass
        extension.extra_compile_args.extend(ccflags)
        ldflags = split_quoted(self['PETSC_EXTERNAL_LIB_BASIC'])
        extension.extra_link_args.extend(ldflags)

    def configure_compiler(self, compiler):
        from distutils.sysconfig import get_config_vars
        if compiler.compiler_type == 'unix':
            (cc, cxx, cflags, ccshared, ldshared, so_ext) = \
            get_config_vars('CC', 'CXX', 'CFLAGS',
                            'CCSHARED', 'LDSHARED', 'SO')
            #
            if self.language == 'c++':
                CXX = self['PCC']
            else:
                try:
                    CXX = self['CXX']
                except KeyError:
                    CXX = cxx
            #
            def extra_flags(cmd):
                cmd  = cmd.split(' ', 1)
                try: return cmd[1]
                except IndexError: return ''
            if self.language == 'c':
                cc_cmd = cc
                ld_cmd = ldshared
            elif self.language == 'c++':
                cflags = cflags.replace('-Wstrict-prototypes','')
                cc_cmd = cxx
                ld_cmd = ldshared
            ccflags = '%s %s %s' % (extra_flags(cc_cmd), cflags, ccshared,)
            ldflags = '%s'       % (extra_flags(ld_cmd),)
            CC_SHARED  = self['PCC']        + ' ' + ccflags
            LD_SHARED  = self['PCC_LINKER'] + ' ' + ldflags
            #
            compiler.set_executables(
                compiler     = self['PCC'],
                compiler_cxx = CXX,
                compiler_so  = CC_SHARED,
                linker_so    = LD_SHARED,
                linker_exe   = self['PCC_LINKER'],
                )
            compiler.shared_lib_extension = so_ext

    def log_info(self):
        log.info('PETSC_DIR:   %s' % self['PETSC_DIR']  )
        log.info('PETSC_ARCH:  %s' % self['PETSC_ARCH'] )
        scalar_type = self['PETSC_SCALAR']
        precision   = self['PETSC_PRECISION']
        language    = self['PETSC_LANGUAGE']
        compiler    = self['PCC']
        linker      = self['PCC_LINKER']
        log.info('scalar-type: %s' % scalar_type)
        log.info('precision:   %s' % precision)
        log.info('language:    %s' % language)
        log.info('compiler:    %s' % compiler)
        log.info('linker:      %s' % linker)


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
        bmake_dir = os.path.join(self.petsc_dir, 'bmake')
        have_bmake = os.path.isdir(bmake_dir)
        log.info('-' * 70)
        log.info('PETSC_DIR:   %s' % self.petsc_dir)
        arch_list = petsc_arch
        if not have_bmake and not arch_list :
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

    @staticmethod
    def get_petsc_dir(petsc_dir):
        if not petsc_dir: return None
        petsc_dir = os.path.expandvars(petsc_dir)
        if not petsc_dir or '$PETSC_DIR' in petsc_dir:
            log.warn("PETSC_DIR not specified")
            return None
        petsc_dir = os.path.expanduser(petsc_dir)
        petsc_dir = os.path.abspath(petsc_dir)
        return config.chk_petsc_dir(petsc_dir)

    @staticmethod
    def chk_petsc_dir(petsc_dir):
        if not os.path.isdir(petsc_dir):
            log.error('invalid PETSC_DIR: %s (ignored)' % petsc_dir)
            return None
        return petsc_dir

    @staticmethod
    def get_petsc_arch(petsc_dir, petsc_arch):
        if not petsc_dir: return None
        petsc_arch = os.path.expandvars(petsc_arch)
        if (not petsc_arch or '$PETSC_ARCH' in petsc_arch):
            have_dir_bmake = os.path.isdir(os.path.join(petsc_dir, 'bmake'))
            have_dir_conf  = os.path.isdir(os.path.join(petsc_dir, 'conf'))
            if have_dir_bmake:
                log.warn("PETSC_ARCH not specified, trying default")
                petscconf = os.path.join(petsc_dir, 'bmake', 'petscconf')
                if not os.path.exists(petscconf):
                    log.warn("file '%s' not found" % petscconf)
                    return None
                conf = StringIO(open(petscconf).read())
                conf = cfgutils.makefile(conf)
                petsc_arch = conf.get('PETSC_ARCH', '')
                if not petsc_arch:
                    log.warn("default PETSC_ARCH not found")
                    return None
            elif have_dir_conf:
                petscvars = os.path.join(petsc_dir, 'conf', 'petscvariables')
                if os.path.exists(petscvars):
                    conf = StringIO(open(petscvars).read())
                    conf = cfgutils.makefile(conf)
                    petsc_arch = conf.get('PETSC_ARCH', '')
            else:
                petsc_arch = ''
        petsc_arch = petsc_arch.split(os.pathsep)
        petsc_arch = cfgutils.unique(petsc_arch)
        petsc_arch = [arch for arch in petsc_arch if arch]
        return config.chk_petsc_arch(petsc_dir, petsc_arch)

    @staticmethod
    def chk_petsc_arch(petsc_dir, petsc_arch):
        have_bmake = os.path.isdir(os.path.join(petsc_dir, 'bmake'))
        valid_archs = []
        for arch in petsc_arch:
            if have_bmake:
                arch_path = os.path.join(petsc_dir, 'bmake', arch)
            else:
                arch_path = os.path.join(petsc_dir, arch)
            if os.path.isdir(arch_path):
                valid_archs.append(arch)
            else:
                log.warn("invalid PETSC_ARCH '%s' (ignored)" % arch)
        if have_bmake and not valid_archs:
            log.warn("could not find any valid PETSC_ARCH")
        return valid_archs


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
        import sys, os
        from distutils import sysconfig
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
        extclass = ext.__class__
        fullname = self.get_ext_fullname(ext.name)
        modpath = str.split(fullname, '.')
        pkgpath = os.path.join('', *modpath[0:-1])
        name = modpath[-1]
        srcs = list(ext.sources)
        macs = list(ext.define_macros)
        incs = list(ext.include_dirs)
        deps = list(ext.depends)
        lang = ext.language
        newext = extclass(name, sources=srcs, depends=deps,
                          define_macros=macs, include_dirs=incs,
                          language=lang)
        return pkgpath, newext

    def _build_ext_arch(self, ext, pkgpath, arch):
        build_temp = self.build_temp
        build_lib  = self.build_lib
        self.build_temp = os.path.join(build_temp, arch)
        self.build_lib  = os.path.join(build_lib, pkgpath, arch)
        _build_ext.build_extension(self, ext)
        self.build_lib  = build_lib
        self.build_temp = build_temp

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
            ext.language = config.language
            config.log_info()
            pkgpath, newext = self._copy_ext(ext)
            config.configure(newext, self.compiler)
            name =  self.distribution.get_name()
            version = self.distribution.get_version()
            distdir = "%s-%s/" % (name, version)
            newext.define_macros.append(('__INSDIR__', distdir))
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
        log.info('writing %s' % config_file)
        fh = open(config_file, 'w')
        try: fh.write(config_data)
        finally: fh.close()

    def get_config_data(self, arch_list):
        template = """\
PETSC_DIR  = %(PETSC_DIR)s
PETSC_ARCH = %(PETSC_ARCH)s
"""
        variables = {'PETSC_DIR'  : self.petsc_dir,
                     'PETSC_ARCH' : os.path.pathsep.join(arch_list)}
        return template, variables

    def get_outputs(self):
        self.check_extensions_list(self.extensions)
        outputs = []
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            if isinstance(ext, Extension):
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


# --------------------------------------------------------------------
