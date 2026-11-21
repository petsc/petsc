import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions                = ['SDL_CreateWindow', 'TTF_Init']
    self.includes                 = ['SDL3/SDL.h', 'SDL3_ttf/SDL_ttf.h']
    self.liblist                  = [['libSDL3_ttf.a', 'libSDL3.a']]
    self.lookforbydefault         = 1
    self.pkgname                  = 'sdl3 sdl3-ttf'
    self.skipMPIDependency        = 1
    return

  def getSearchDirectories(self):
    '''Generate list of possible locations of SDL3 and SDL3_ttf'''
    yield ''
    yield '/usr/local'
    yield '/opt/homebrew'
    yield '/opt/local'
    yield '/usr'
    return
