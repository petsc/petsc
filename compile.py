#!/usr/bin/env python
import action
import fileset

class CompileSIDL (action.Action):
  def __init__(self, generatedSources, sources = None, compiler = 'babel', compilerFlags = '-sC++ -ogenerated'):
    action.Action.__init__(self, compiler, sources, '--suppress-timestamp '+compilerFlags, 1)
    self.generatedSources = generatedSources

  def shellSetAction(self, set):
    if not set.tag == 'unchanged':
      action.Action.shellSetAction(self, set)
    self.products = self.generatedSources
