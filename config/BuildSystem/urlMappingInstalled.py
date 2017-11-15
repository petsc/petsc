import urlparse
# Fix parsing for nonstandard schemes
urlparse.uses_netloc.extend(['hg', 'ssh'])

def installedUrlMap(self, url):
  import os
  project = self.getInstalledProject(url)
  if project:
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    path = project.getRoot()
    return (1, urlparse.urlunparse(('hgfile', '', path, parameters, query, fragment)))
  return (0, url)

def setupUrlMapping(self, urlMaps):
  urlMaps.insert(0, lambda url, self = self: installedUrlMap(self, url))
  return
