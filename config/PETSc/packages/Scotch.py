#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['http://gforge.inria.fr/frs/download.php/10715/scotch_5.1.2.tar.gz']
    self.downloadname = self.name.lower()
    self.liblist      = [['libscotch.a','libscotcherr.a'],
                         ['libscotch.a','libscotcherr.a','librt.a']]
    self.functions    = ['SCOTCH_archBuild']
    self.includes     = ['scotch.h']
    self.complex      = 0
    self.needsMath    = 1
    return

  def __str__(self):
    if self.found:
      desc = ['Scotch:']	
      desc.append('  Version: '+self.version)
      desc.append('  Includes: '+str(self.include))
      desc.append('  Library: '+str(self.lib))
      return '\n'.join(desc)+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('Scotch', '-with-scotch=<bool>',                nargs.ArgBool(None, 0, 'Activate Scotch'))
    help.addArgument('Scotch', '-with-scotch-dir=<root dir>',        nargs.ArgDir(None, None, 'Specify the root directory of the Scotch installation'))
    help.addArgument('Scotch', '-download-scotch=<no,yes,ifneeded>', nargs.ArgFuzzyBool(None, 0, 'Automatically install Scotch'))
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.libraryOptions = framework.require('PETSc.utilities.libraryOptions', self)
    self.deps       = [self.mpi]
    return

  def Install(self):

    self.logPrintBox('Creating Scotch '+os.path.join(os.path.join(self.packageDir,'src'),'Makefile.inc')+'\n')
    g = open(os.path.join(os.path.join(self.packageDir,'src'),'Makefile.inc'),'w')

    g.write('EXE	=\n')
    g.write('LIB	= .a\n')
    g.write('OBJ	= .o\n')
    g.write('\n')
    g.write('MAKE	= make\n')

    g.write('AR	        = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS	= '+self.setCompilers.AR_FLAGS+'\n')
    g.write('CAT	= cat\n')   
    self.setCompilers.pushLanguage('C')
    g.write('CC	        = '+self.setCompilers.getCompiler()+'\n')
    g.write('CCP        = '+self.setCompilers.getCompiler()+'\n')
   
    # Building cflags
    self.cflags = self.setCompilers.getCompilerFlags()
    if self.libraries.add('-lz','gzwrite'): 
      self.cflags = self.cflags + ' -DCOMMON_FILE_COMPRESS_GZ'
    self.cflags = self.cflags + ' -DCOMMON_PTHREAD -DCOMMON_RANDOM_FIXED_SEED' 
    self.cflags = self.cflags + ' -DSCOTCH_PTHREAD -DSCOTCH_RENAME '
    # this is needed on the Mac, because common2.c includes common.h which DOES NOT include mpi.h because
    # SCOTCH_PTSCOTCH is NOT defined above Mac does not know what clock_gettime() is!
    if self.setCompilers.isDarwin():    
      self.cflags = self.cflags + ' -DCOMMON_TIMING_OLD'

    if self.libraryOptions.integerSize == 64:
      self.cflags = self.cflags + ' -DINTSIZE64'
    else:
      self.cflags = self.cflags + ' -DINTSIZE32'
    g.write('CFLAGS	= '+self.cflags+'\n')

    self.setCompilers.popLanguage()
    ldflags = ''
    if self.libraries.add('-lz','gzwrite'): ldflags += '-lz'
    if self.libraries.add('-lm','sin'): ldflags += ' -lm'
    if self.libraries.add('-lrt','timer_create'): ldflags += ' -lrt'
    g.write('LDFLAGS	= '+ldflags+'\n')
    g.write('CP         = '+self.programs.cp+'\n')
    g.write('LEX	= flex\n')
    g.write('LN	        = ln\n')
    g.write('MKDIR      = '+self.programs.mkdir+'\n')
    g.write('MV         = '+self.programs.mv+'\n')
    g.write('RANLIB	= '+self.setCompilers.RANLIB+'\n')
    g.write('YACC	= bison -y\n')
    g.close()

    if self.installNeeded(os.path.join('src','Makefile.inc')):
      try:
        self.logPrintBox('Compiling Scotch; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && make clean scotch', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on Scotch: '+str(e))
#      try:
#        output = config.base.Configure.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+'; ls libscotch; make scotch',timeout=2500, log = self.framework.log)[0]
      libDir     = os.path.join(self.installDir, self.libdir)
      includeDir = os.path.join(self.installDir, self.includedir)
      output = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; cp -f lib/*.a '+libDir+'/.; cp -f include/*.h '+includeDir+'/.;', timeout=2500, log = self.framework.log)[0]
#      except RuntimeError, e:
#        raise RuntimeError('Error running make on Scotch: '+str(e))
      self.checkInstall(output,os.path.join('src','Makefile.inc'))
    return self.installDir


if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
