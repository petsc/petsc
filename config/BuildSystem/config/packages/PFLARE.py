import config.package

class Configure(config.package.Package):
    def __init__(self, framework):
        config.package.Package.__init__(self, framework)
        self.version                = '1.26.0'
        self.gitcommit              = '1faebf6106d65bdda021043db663652788534793' # main branch as of 2026-07-12
        self.download               = ['git://https://github.com/PFLAREProject/PFLARE','https://github.com/PFLAREProject/PFLARE/archive/'+self.gitcommit+'.tar.gz']
        self.functions              = ['PCRegister_PFLARE']
        self.includes               = ['pflare.h']
        self.providesDocs           = 1                 # docs pipeline should clone + scan this package's sources for manual pages
        self.docsDirs               = ['src','include'] # subdirs that contain the manual pages - /*MC*/, /*@ @*/, /*E*/ blocks
        self.liblist                = [['libpflare.a']]
        self.complex                = 0
        self.precisions             = ['double']
        self.linkedbypetsc          = 0
        self.builtafterpetsc        = 1
        return

    def setupDependencies(self, framework):
        config.package.Package.setupDependencies(self, framework)
        self.sharedLibraries = framework.require('PETSc.options.sharedLibraries',self)
        self.mpi             = framework.require('config.packages.MPI',self)
        self.blasLapack      = framework.require('config.packages.BlasLapack',self)
        self.parmetis        = framework.require('config.packages.ParMETIS',self)
        self.kokkos          = framework.require('config.packages.kokkos',self)
        self.kokkoskernels   = framework.require('config.packages.kokkos-kernels',self)
        self.scalartypes     = framework.require('PETSc.options.scalarTypes',self)
        self.petsc4py        = framework.require('config.packages.petsc4py',self)
        self.cython          = framework.require('config.packages.Cython',self)
        self.deps            = [self.blasLapack]
        self.odeps           = [self.mpi,self.parmetis,self.kokkos,self.kokkoskernels,self.petsc4py,self.cython]
        return

    def Install(self):
        import os
        # Have to be built with shared libraries if using the PETSc configure
        # For static builds you have to build PFLARE yourself from source
        if not self.checkSharedLibrariesEnabled():
            raise RuntimeError('PFLARE built through the PETSc configure requires shared libraries')

        # if installing prefix location then need to set new value for PETSC_DIR/PETSC_ARCH
        if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
            barg = ' PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH=""'
            prefix = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
        else:
            barg = 'PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH}'
            prefix = os.path.join(self.petscdir.dir,self.arch)
        barg = barg + ' PREFIX=' + prefix
        checkarg = barg

        self.include = [os.path.join(self.installDir,'include')]
        self.lib     = [os.path.join(self.installDir,'lib','libpflare.a')]
        libdir = os.path.join(self.installDir, 'lib')
        # Call make, make python (if petsc4py is enabled), then
        # make install. After that create a symlink called libpetscpflare
        # to ensure the PFLARE registration routine is called when PETSc loads
        # the dynamic PFLARE library
        post_cmds = [barg + ' ${OMAKE} ' + barg,]
        if self.petsc4py.found:
            if not self.cython.found:
                raise RuntimeError('PFLARE with petsc4py requires cython! Suggest --download-cython!')
            post_cmds.append(barg + ' ${OMAKE} ' + barg + ' python')
        post_cmds.append(barg + ' ${OMAKE} ' + barg + ' install')
        post_cmds.append('cd "{0}" && ln -sf libpflare.{1} libpetscpflare.{1}'.format(libdir, self.setCompilers.sharedLibraryExt))
        self.addPost(self.packageDir, post_cmds)
        self.addMakeCheck(self.packageDir, '${OMAKE} ' + checkarg + ' check')

        return self.installDir
