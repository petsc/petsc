#!/usr/bin/env python
import fileset
import logging
import transform

import types

class Target (transform.Transform):
  "Targets are entry points into the build process, and provide higher level control flow"
  def __init__(self, sources = None, transforms = []):
    transform.Transform.__init__(self, sources)
    self.transforms = transforms[:]

  def executeSingleTransform(self, sources, transform):
    self.debugPrint('Executing transform '+str(transform)+' with sources '+self.debugFileSetStr(sources), 1, 'target')
    if len(transform.sources):
      if isinstance(transform.sources, fileset.FileSet):
        if isinstance(sources, fileset.FileSet):
          transform.sources = [transform.sources, sources]
        else:
          transform.sources = [transform.sources]+sources
      else:
        if isinstance(sources, fileset.FileSet):
          transform.sources.append(sources)
        else:
          transform.sources = transform.sources+sources
    else:
      transform.sources = sources
    products = transform.execute()
    self.debugPrint('Transform products '+self.debugFileSetStr(products), 4, 'target')
    return products

  def executeTransformPipe(self, sources, list):
    for transform in list:
      products = self.executeTransform(sources, transform)
      sources  = products
    return products

  def executeTransformFan(self, sources, tuple):
    products = []
    for transform in tuple:
      p = self.executeTransform(sources, transform)
      if type(p) == types.ListType:
        products.extend(p)
      else:
        products.append(p)
    return products

  def executeTransform(self, sources, t):
    if isinstance(t, transform.Transform):
      products = self.executeSingleTransform(sources, t)
    elif isinstance(t, types.ListType):
      products = self.executeTransformPipe(sources, t)
    elif isinstance(t, types.TupleType):
      products = self.executeTransformFan(sources, t)
    else:
      raise RuntimeError('Invalid transform type '+type(t))
    self.debugPrint('Target products '+self.debugFileSetStr(products), 2, 'target')
    return products

  def execute(self):
    return self.executeTransform(self.sources, self.transforms)
