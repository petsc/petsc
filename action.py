#!/usr/bin/env python
import fileset
import transform

class Action (transform.Transform):
  def __init__(self, func, sources = None, flags = '', setwiseExecute = 1, errorHandler = None):
    transform.Transform.__init__(self, sources)
    if callable(func):
      self.func         = func
      if (setwiseExecute):
        self.setExecute = self.setAction
    else:
      self.func         = self.shellAction
      self.program      = func
      if (setwiseExecute):
        self.setExecute = self.shellSetAction
    self.flags          = flags
    self.setwiseExecute = setwiseExecute
    self.errorHandler   = errorHandler
    self.buildProducts  = 1

  def constructFlags(self, source, baseFlags):
    return baseFlags

  def fileExecute(self, file):
    self.func(file)
    if self.buildProducts: self.products.append(file)

  def setAction(self, set):
    self.func(set)
    if self.buildProducts:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products, set]
      else:
        self.products.append(set)

  def shellAction(self, file):
    command = self.program+' '+self.constructFlags(file, self.flags)+' '+file
    output  = self.executeShellCommand(command, self.errorHandler)
    if self.buildProducts: self.products.append(file)
    return output

  def shellSetAction(self, set):
    files   = set.getFiles()
    if (not files): return ''
    command = self.program+' '+self.constructFlags(set, self.flags)
    for file in files:
      command += ' '+file
    output  = self.executeShellCommand(command, self.errorHandler)
    if self.buildProducts:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products, set]
      else:
        self.products.append(set)
    return output
