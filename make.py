#!/usr/bin/env python
import bs
import project

class PetscMake(bs.BS):
  def __init__(self, clArgs = None, argDB = None):
    bs.BS.__init__(self, project.Project('bs', 'bk://sidl.bkbits.net/BuildSystem', self.getRoot()), clArgs, argDB)
    return

  def setupBuild(self):
    sidl = self.getSIDLDefaults()
    sidl.usingSIDL.requiresRuntime = 0
    sidl.addClientLanguage('Python')
    return

if __name__ ==  '__main__':
  import sys
  pm = PetscMake(sys.argv[1:])
  pm.main()
