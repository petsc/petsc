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

  def defineHelp(self):
    bs.argDB.setHelp('PYTHON_INCLUDE', 'The directory in which the Python headers were installed (like Python.h)')
    bs.argDB.setHelp('BABEL_DIR', 'The directory in which Babel was installed')

  def defineDirectories(self):
    self.directories['sidl']               = os.path.join(os.getcwd(), 'sidl')
    self.directories['serverSource']       = os.path.join(os.getcwd(), 'server')
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
    self.filesets['serverSource']       = fileset.ExtensionFileSet(self.directories['serverSource'], ['.h', '.c', '.hh', '.cc'])
    self.filesets['pythonClientSource'] = fileset.ExtensionFileSet(self.directories['pythonClientSource'], ['.h', '.c'])
    self.filesets['serverLib']          = fileset.FileSet([os.path.join(self.directories['lib'], 'libbs.a')])
    self.filesets['pythonClientLib']    = fileset.FileSet([os.path.join(self.directories['lib'], 'libpythonbs.a')])

  def defineTargets(self):
    sidlRepositoryAction = babel.CompileSIDLRepository()
    sidlServerAction = babel.CompileSIDLServer(self.filesets['serverSource'])
    sidlServerAction.outputDir = self.directories['serverSource']
    sidlPythonClientAction = babel.CompileSIDLClient()
    sidlPythonClientAction.outputDir = self.directories['pythonClientSource']
    serverCAction = compile.CompileC(self.filesets['serverLib'])
    serverCAction.defines.append('PIC')
    serverCAction.includeDirs.append(self.directories['babelInc'])
    serverCxxAction = compile.CompileCxx(self.filesets['serverLib'])
    serverCxxAction.defines.append('PIC')
    serverCxxAction.includeDirs.append(self.directories['babelInc'])
    serverCxxAction.includeDirs.append(self.directories['serverSource'])
    pythonClientCAction = compile.CompileC(self.filesets['pythonClientLib'])
    pythonClientCAction.defines.append('PIC')
    pythonClientCAction.includeDirs.append(self.directories['babel'])
    pythonClientCAction.includeDirs.append(self.directories['babelInc'])
    pythonClientCAction.includeDirs.append(self.directories['babelPythonInc'])
    pythonClientCAction.includeDirs.append(self.directories['serverSource'])
    pythonClientCAction.includeDirs.append(self.directories['pythonInc'])

    self.targets['sidl']  = target.Target(self.filesets['sidl'],
                                          [babel.TagSIDL(),
                                           bk.TagBKOpen(root = self.directories['serverSource']),
                                           bk.BKOpen(),
                                           (sidlRepositoryAction,
                                            sidlServerAction,
                                            sidlPythonClientAction),
                                           bk.TagBKClose(root = self.directories['serverSource']),
                                           transform.FileFilter(self.isImpl, tags = 'bkadd'),
                                           bk.BKClose()])
    self.targets['serverCompile'] = target.Target(None,
                                                  [self.targets['sidl'],
                                                   compile.TagC(),
                                                   compile.TagCxx(),
                                                   serverCAction,
                                                   serverCxxAction,
                                                   link.TagLibrary(),
                                                   link.LinkSharedLibrary(extraLibraries=self.filesets['babelLib'])])
    self.targets['pythonClientCompile'] = target.Target(self.filesets['pythonClientSource'],
                                                  [compile.TagC(),
                                                   transform.FileFilter(self.isPythonStub, tags = ['c', 'old c']),
                                                   pythonClientCAction,
                                                   link.TagLibrary(),
                                                   link.LinkSharedLibrary(extraLibraries=self.filesets['babelLib'])])
    self.targets['pythonModuleFixup'] = target.Target(fileset.ExtensionFileSet(self.directories['pythonClientSource'], '.c'),
                                                      [babel.PythonModuleFixup(self.filesets['pythonClientLib'], self.directories['pythonClientSource'])])

  def isImpl(self, source):
    if self.implRE.match(os.path.dirname(source)): return 1
    return 0

  def isPythonStub(self, source):
    if os.path.dirname(source) == self.directories['pythonClientSource']: return 0
    return 1

if __name__ ==  '__main__': PetscMake(sys.argv[1:]).main()
