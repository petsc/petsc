#!/usr/bin/env python
import babel
import bk
import bs
import compile
import fileset
import link
import target
import transform

import os
import re
import sys

class PetscMake(bs.BS):
  implRE = re.compile(r'^(.*)_Impl$')

  def __init__(self, args = None):
    bs.BS.__init__(self, args)
    self.defineHelp()
    self.defineDirectories()
    self.defineFileSets()
    self.defineTargets()
    self.setupDirs()

  def setupDirs(self):
    for root in self.filesets['serverSourceRoots']:
      if not os.path.exists(root):
        os.mkdir(root)

  def defineHelp(self):
    bs.argDB.setHelp('PYTHON_INCLUDE', 'The directory in which the Python headers were installed (like Python.h)')
    bs.argDB.setHelp('BABEL_DIR', 'The directory in which Babel was installed')

  def defineDirectories(self):
    self.directories['sidl']               = os.path.join(os.getcwd(), 'sidl')
    self.directories['sidlRepository']     = os.path.join(os.getcwd(), 'xml')
    self.directories['serverSourceBase']   = os.path.join(os.getcwd(), 'server')
    self.directories['pythonClientSource'] = os.path.join(os.getcwd(), 'python')
    self.directories['lib']                = os.path.join(os.getcwd(), 'lib')
    self.directories['pythonInc']          = bs.argDB['PYTHON_INCLUDE']
    self.directories['babel']              = bs.argDB['BABEL_DIR']
    self.directories['babelInc']           = os.path.join(self.directories['babel'], 'include')
    self.directories['babelPythonInc']     = os.path.join(self.directories['babel'], 'python')
    self.directories['babelLib']           = os.path.join(self.directories['babel'], 'lib')

  def defineFileSets(self):
    self.filesets['babelLib']           = fileset.FileSet([os.path.join(self.directories['babelLib'], 'libsidl.so')])

    self.filesets['sidl']               = fileset.ExtensionFileSet(self.directories['sidl'], '.sidl')
    self.filesets['serverSourceRoots']  = fileset.FileSet(map(lambda file, dir=self.directories['serverSourceBase']:
                                                              dir+'-'+os.path.splitext(os.path.split(file)[1])[0],
                                                              self.filesets['sidl']))
    self.filesets['pythonClientSource'] = fileset.ExtensionFileSet(self.directories['pythonClientSource'], ['.h', '.c'])
    self.filesets['serverLib']          = fileset.FileSet([os.path.join(self.directories['lib'], 'libbs.a')])
    self.filesets['pythonClientLib']    = fileset.FileSet([os.path.join(self.directories['lib'], 'libpythonbs.a')])

  def defineTargets(self):
    sidlRepositoryAction = babel.CompileSIDLRepository()
    sidlCxxServerAction = babel.CompileSIDLServer(None)
    sidlCxxServerAction.outputDir = self.directories['serverSourceBase']
    sidlCxxServerAction.repositoryDirs.append(self.directories['sidlRepository'])
    sidlPythonClientAction = babel.CompileSIDLClient()
    sidlPythonClientAction.language  = 'Python'
    sidlPythonClientAction.outputDir = self.directories['pythonClientSource']
    sidlPythonClientAction.repositoryDirs.append(self.directories['sidlRepository'])

    pythonClientCAction = compile.CompileC(self.filesets['pythonClientLib'])
    pythonClientCAction.defines.append('PIC')
    pythonClientCAction.includeDirs.append(self.directories['pythonClientSource'])
    pythonClientCAction.includeDirs.append(self.directories['babelInc'])
    pythonClientCAction.includeDirs.append(self.directories['babelPythonInc'])
    pythonClientCAction.includeDirs.append(self.directories['pythonInc'])

    serverTargets = []
    for sidlFile in self.filesets['sidl'].getFiles():
      package   = os.path.splitext(os.path.split(sidlFile)[1])[0]
      rootDir   = self.directories['serverSourceBase']+'-'+package
      library   = fileset.FileSet([os.path.join(self.directories['lib'], 'libserver-'+package+'.a')])
      libraries = self.filesets['babelLib']

      serverCAction = compile.CompileC(library)
      serverCAction.defines.append('PIC')
      serverCAction.includeDirs.append(self.directories['babelInc'])

      serverCxxAction = compile.CompileCxx(library)
      serverCxxAction.defines.append('PIC')
      serverCxxAction.includeDirs.append(self.directories['babelInc'])
      serverCxxAction.includeDirs.append(self.directories['serverSourceBase']+'-'+package)

      serverTargets.append(target.Target(fileset.ExtensionFileSet(rootDir, ['.h', '.c', '.hh', '.cc']),
                                         [compile.TagC(root = rootDir),
                                          compile.TagCxx(root = rootDir),
                                          serverCAction,
                                          serverCxxAction,
                                          link.TagLibrary(),
                                          link.LinkSharedLibrary(extraLibraries = libraries)]))

    self.targets['repositorySidl']  = target.Target(None,
                                                    [babel.TagAllSIDL(),
                                                     sidlRepositoryAction])
    self.targets['cxxServerSidl']  = target.Target(None,
                                                   [bk.TagBKOpen(roots = self.filesets['serverSourceRoots']),
                                                    bk.BKOpen(),
                                                    babel.TagSIDL(),
                                                    sidlCxxServerAction,
                                                    bk.TagBKClose(roots = self.filesets['serverSourceRoots']),
                                                    transform.FileFilter(self.isImpl, tags = 'bkadd'),
                                                    bk.BKClose()])
    self.targets['pythonClientSidl']  = target.Target(None,
                                                      [babel.TagAllSIDL(),
                                                       sidlPythonClientAction])
    self.targets['sidl']  = target.Target(self.filesets['sidl'],
                                          [(self.targets['repositorySidl'],
                                            self.targets['cxxServerSidl'],
                                            self.targets['pythonClientSidl']),
                                           transform.Update()
                                           ])

    self.targets['serverCompile'] = target.Target(None,
                                                  [tuple(serverTargets),
                                                   transform.Update()])

    self.targets['pythonModuleFixup'] = target.Target(None,
                                                      [transform.FileFilter(lambda source: source[-2:] == '.c'),
                                                       babel.PythonModuleFixup(self.filesets['pythonClientLib'], self.directories['pythonClientSource'])])
    self.targets['pythonClientCompile'] = target.Target(self.filesets['pythonClientSource'],
                                                        (self.targets['pythonModuleFixup'],
                                                         [compile.TagC(),
                                                          pythonClientCAction,
                                                          link.TagLibrary(),
                                                          link.LinkSharedLibrary(extraLibraries=self.filesets['babelLib'])]))

    self.targets['default'] = target.Target(None,
                                            [(self.targets['sidl'], self.targets['serverCompile'], self.targets['pythonClientCompile'])])

  def isImpl(self, source):
    if self.implRE.match(os.path.dirname(source)): return 1
    return 0

if __name__ ==  '__main__': PetscMake(sys.argv[1:]).main()
