#!/usr/bin/env python
import user
import base

import os

class RemoteBuild (base.Base):
  def __init__(self, clArgs = None, argDB = None):
    base.Base.__init__(self, clArgs, argDB)
    self.dir  = os.path.join('/sandbox', 'petsc', 'petsc-3')
    self.rsh  = 'ssh -1'
    self.rcp  = 'scp -q -B -oProtocol=1'
    self.user = 'petsc'
    self.host = 'harley.mcs.anl.gov'
    return

  def setupArgDB(self, argDB, clArgs):
    '''Setup argument types, using the database created by base.Base'''
    import nargs

    argDB.setType('dryRun', nargs.ArgBool(None, 0, 'Display but do not execute commands', isTemporary = 1), forceLocal = 1)

    self.argDB['debugLevel']    = 3
    self.argDB['debugSections'] = []

    base.Base.setupArgDB(self, argDB, clArgs)
    return argDB

  def executeShellCommand(self, command, checkCommand = None):
    '''Execute a shell command returning the output, and optionally provide a custom error checker'''
    if self.argDB['dryRun']:
      print command
      return
    return base.Base.executeShellCommand(self, command, checkCommand)

  def clean(self):
    '''Remove all PETSc 3 files'''
    command = [self.rsh, self.user+'@'+self.host, '-n', 'rm -rf '+self.dir]
    output  = self.executeShellCommand(' '.join(command))
    command = [self.rsh, self.user+'@'+self.host, '-n', 'mkdir -m775 '+self.dir]
    output  = self.executeShellCommand(' '.join(command))
    return

  def getBootstrap(self):
    '''Right now, we get bootstrap.py from our PETSc 2 repository, but later we should get it from the webpage'''
    command = [self.rsh, self.user+'@'+self.host, '-n', 'cp', os.path.join('/sandbox', 'petsc', 'petsc-test', 'python', 'BuildSystem', 'install', 'bootstrap.py'), self.dir]
    output  = self.executeShellCommand(' '.join(command))
    return

  def bootstrap(self):
    '''Run the bootstrap installer
       - TODO: Remove the dependence on csh of the pipe'''
    self.getBootstrap()
    command = [self.rsh, self.user+'@'+self.host, '-n', '"cd '+self.dir+'; ./bootstrap.py -batch |& tee bootstrap.log"']
    output  = self.executeShellCommand(' '.join(command))
    return

  def install(self, package, args = []):
    '''Install a normal package'''
    command = [self.rsh, self.user+'@'+self.host, '-n', '"cd '+self.dir+';', os.path.join('.', 'BuildSystem', 'install', 'installer.py'), package]+args+['|& tee installer.log"']
    output  = self.executeShellCommand(' '.join(command))
    return

  def run(self):
    self.clean()
    self.bootstrap()
    self.install('bk://mpib.bkbits.net/mpib-dev', ['--with-mpi-dir=/home/petsc/soft/linux-rh73/mpich-1.2.4'])
    return

  def copyLog(self):
    '''Copy all logs made during the build to a default location'''
    command = [self.rcp, self.user+'@'+self.host+':'+os.path.join(self.dir, 'bootstrap.log'), os.path.join('/home', 'petsc', 'logs', 'nightly')]
    output  = self.executeShellCommand(' '.join(command))
    command = [self.rcp, self.user+'@'+self.host+':'+os.path.join(self.dir, 'installer.log'), os.path.join('/home', 'petsc', 'logs', 'nightly')]
    output  = self.executeShellCommand(' '.join(command))
    command = [self.rcp, self.user+'@'+self.host+':'+os.path.join(self.dir, 'make.log'), os.path.join('/home', 'petsc', 'logs', 'nightly')]
    output  = self.executeShellCommand(' '.join(command))
    return not status

if __name__ == '__main__':
  import sys
  RemoteBuild(sys.argv[1:]).run()
