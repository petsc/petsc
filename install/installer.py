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

  def install(self, projectUrl):
    self.debugPrint('Installing '+projectUrl, 3, 'install')
    root = self.retriever.retrieve(projectUrl);
    self.builder.build(root)
    return

if __name__ == '__main__':
  installer = Installer(sys.argv[1:])
  if installer.argDB.has_key('projectUrl'):
    url = installer.argDB['projectUrl']
  else:
    url = 'bk://sidl.bkbits.net/Runtime'
  installer.install(url)
