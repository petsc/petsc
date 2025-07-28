import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions = ['glFlush','gluBeginCurve']
    self.includes = ['GL/gl.h','GL/glu.h']
    self.liblist = [['libGL.a','libGLU.a']]
    return

  def getSearchDirectories(self):
    '''Generate list of possible locations of OpenGL'''
    yield ''
    yield '/opt/X11'
    yield '/usr/X11'
    yield '/usr/X11R6'
    yield '/usr/X11R5'
    yield '/usr/X11R4'
    yield '/usr/local/X11'
    yield '/usr/local/X11R6'
    yield '/usr/local/X11R5'
    yield '/usr/local/X11R4'
    yield '/usr/X386'
    yield '/usr/x386'
    yield '/usr/XFree86/X11'
    yield '/usr/local'
    yield '/usr/local/x11r5'
    yield '/usr/lpp/Xamples'
    yield '/usr/openwin'
    yield '/usr/openwin/share'
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = []
    return
