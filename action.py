#!/usr/bin/env python
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

  def fileExecute(self, file):
    self.func(file)
    self.products.append(file)

  def setAction(self, set):
    self.func(set)
    self.products.extend(set)

  def shellAction(self, file):
    command = self.program+' '+self.flags+' '+file
    output  = self.executeShellCommand(command, self.errorHandler)
    self.products.append(file)
    return output

  def shellSetAction(self, set):
    command = self.program+' '+self.flags
    files   = set.getFiles()
    if (not files): return ''
    for file in files:
      command += ' '+file
    output  =  self.executeShellCommand(command, self.errorHandler)
    self.products.extend(set)
    return output
