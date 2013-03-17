import urlparse
# Fix parsing for nonstandard schemes
urlparse.uses_netloc.extend(['hg', 'ssh'])

def bootstrapUrlMap(self, url):
  import os
  if self.checkBootstrap():
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    if scheme == 'hg':
      path = os.path.join('/tmp', self.getRepositoryPath(url))
      return (1, urlparse.urlunparse(('file', '', path, parameters, query, fragment)))
  return (0, url)

def setupUrlMapping(self, urlMaps):
  urlMaps.insert(0, lambda url, self = self: bootstrapUrlMap(self, url))
  return
