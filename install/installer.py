#!/usr/bin/env python
import install.build
import install.retrieval
import logging
import nargs

import sys

class Installer(logging.Logger):
  def __init__(self, clArgs = None):
    self.argDB     = nargs.ArgDict('ArgDict', clArgs)
    self.retriever = install.retrieval.Retriever(self.argDB)
    self.builder   = install.build.Builder(self.argDB)
    return

  def install(self):
    project = 'bk://sidl.bkbits.net/Runtime'
    self.debugPrint('Installing '+project, 3, 'install')
    root = self.retriever.retrieve(project);
    self.builder.build(root)
    return

if __name__ == '__main__': Installer(sys.argv[1:]).install()
