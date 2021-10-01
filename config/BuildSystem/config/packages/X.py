import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.versionname      = 'XORG_VERSION_MAJOR.XORG_VERSION_MINOR.XORG_VERSION_PATCH.XORG_VERSION_SNAP'
    self.versioninclude   = ['xorg/xorg-server.h','xorg/xorgVersion.h']
    self.functions        = ['XSetWMName']
    self.includes         = ['X11/Xlib.h']
    self.liblist          = [['libX11.a']]
    self.lookforbydefault = 1
    self.pkgname          = 'x11'
    return

  def versionToStandardForm(self,ver):
    '''Completes the arithmetic needed to compute the version number from the numerical strings'''
    return '.'.join([str(eval(i)) for i in ver.split('.')])

  def getSearchDirectories(self):
    '''Generate list of possible locations of X11'''
    if self.setCompilers.isDarwin(self.log):
      yield '/opt/X11'
    yield ''
    if not self.setCompilers.isDarwin(self.log):
      yield '/opt/X11'
    yield '/Developer/SDKs/MacOSX10.5.sdk/usr/X11'
    yield '/Developer/SDKs/MacOSX10.4u.sdk/usr/X11R6'
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
