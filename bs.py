#!/usr/bin/env python
import commands
import os.path
import string

echo = 1

class Maker:
  def executeShellCommand(self, command):
    if echo: print command
    (status, output) = commands.getstatusoutput(command)
    if status: raise IOError('Could not execute \''+command+'\': '+output)
    return output

class Precondition (Maker):
  # True here means that the precondition was satisfied
  def __nonzero__(self):
    if echo: print 'Checking precondition'
    return 1

class Action (Maker):
  # A nonzero return means that the action was successful
  def execute(self):
    if echo: print 'Executing action'
    return 1

class Target (Maker):
  preconditions = []
  actions       = []

  def __init__(self, preconditions=None, actions=None):
    if preconditions: self.preconditions = preconditions
    if actions:       self.actions       = actions

  def execute(self):
    if (reduce(lambda a,b: a and b, self.preconditions)):
      map(lambda x: x.execute(), self.actions)

class OlderThan (Precondition):
  def __init__(self, target, sources=None):
    self.target  = target
    self.sources = sources

  def __nonzero__(self):
    targetTime = os.path.getmtime(self.target)
    if (callable(self.sources)):
      files = self.sources()
    else:
      files = self.sources
    if (not len(files)): return 1
    for source in files:
      sourceTime = os.path.getmtime(source)
      if (targetTime > sourceTime):
        print self.target+' is newer than '+source
        return 1
    return 0

class CompileFiles (Action):
  def __init__(self, compiler, compilerFlags, sources):
    self.compiler      = compiler
    self.compilerFlags = compilerFlags
    self.sources       = sources

  def execute(self):
    print 'Compiling '+str(self.sources)
    #command = self.compiler+' '+self.compilerFlags
    #for source in self.sources: command += ' '+source
    #self.executeShellCommand(command)
