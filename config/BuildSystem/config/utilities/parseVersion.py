import config.base

import re
import functools

@functools.total_ordering
class FallbackVersion:
  '''Stand-in for packaging.version.Version implementing the subset of PEP 440 that PETSc package
     version strings use; provides .release and the comparison operators'''
  regex = re.compile(r"""^\s*v?
      (?:(?P<epoch>[0-9]+)!)?
      (?P<release>[0-9]+(?:\.[0-9]+)*)
      (?:[-_.]?(?P<pre_l>alpha|beta|preview|pre|rc|a|b|c)[-_.]?(?P<pre_n>[0-9]+)?)?
      (?:-(?P<post_n1>[0-9]+)|[-_.]?(?P<post_l>post|rev|r)[-_.]?(?P<post_n2>[0-9]+)?)?
      (?P<dev>[-_.]?dev[-_.]?(?P<dev_n>[0-9]+)?)?
      (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?
      \s*$""", re.VERBOSE | re.IGNORECASE)
  preNames = {'alpha': 'a', 'beta': 'b', 'c': 'rc', 'pre': 'rc', 'preview': 'rc'}

  def __init__(self, version):
    match = self.regex.match(version)
    if not match: raise ValueError('Invalid version: ' + repr(version))
    self.release = tuple(int(i) for i in match.group('release').split('.'))
    epoch        = int(match.group('epoch')) if match.group('epoch') else 0
    preLetter    = match.group('pre_l')
    preNumber    = match.group('pre_n')
    postNumber   = match.group('post_n1') or match.group('post_n2')
    local        = match.group('local')
    postValue    = None
    devValue     = None

    if postNumber:
      postValue = int(postNumber)
    elif match.group('post_l'):
      postValue = 0
    if match.group('dev'): devValue = int(match.group('dev_n')) if match.group('dev_n') else 0
    # trailing zeros are insignificant, so that 1.2 and 1.2.0 compare equal
    releaseKey = list(self.release)
    while releaseKey and releaseKey[-1] == 0: releaseKey.pop()
    if preLetter:
      preKey = (0, self.preNames.get(preLetter.lower(), preLetter.lower()), int(preNumber) if preNumber else 0)
    elif devValue is not None and postValue is None:
      preKey = (-1,) # a dev release with no pre-release segment sorts before everything else
    else:
      preKey = (1,)  # a final release sorts after any pre-release of the same release
    postKey  = (-1,) if postValue is None else (postValue,)
    devKey   = (1,) if devValue is None else (0, devValue)
    localKey = () if local is None else tuple((int(s), '') if s.isdigit() else (-1, s.lower()) for s in re.split(r'[-_.]', local))
    self.key = (epoch, tuple(releaseKey), preKey, postKey, devKey, localKey)
    return

  def __eq__(self, other):
    return self.key == other.key

  def __lt__(self, other):
    return self.key < other.key

  def __hash__(self):
    return hash(self.key)

def loadParseVersion():
  '''Returns packaging.version.parse(); packaging is not part of the Python standard library and
     releases before 22.0 return a LegacyVersion (which has no .release and orders differently)
     instead of raising on invalid input, so fall back to FallbackVersion in both cases'''
  try:
    from packaging.version import parse
    parse('not a version')
  except ImportError:
    pass
  except Exception:
    return parse
  return FallbackVersion

parseVersion = loadParseVersion()

class Configure(config.base.Configure):
  '''Has no configure tests; PETSc/Configure.py registers every file in this directory as a child,
     so this module must provide a Configure class even though it only exports parseVersion()'''
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str__(self):
    return ''
