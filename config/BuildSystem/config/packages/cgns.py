#!/usr/bin/env python
import config.base
import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions  = ['cg_close']
    self.includes   = ['cgnslib.h']
    self.liblist    = [['libcgns.a'],
                       ['libcgns.a', 'libhdf5.a']] # Sometimes they over-link
