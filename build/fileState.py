import build.fileset
import build.transform

import os

class FileChanged (build.transform.Transform):
  '''Detects whether files have changed using checksums
     - If the force flag is given, all files are marked changed'''
  def __init__(self, sourceDB, inputTag = None, changedTag = 'changed', unchangedTag = 'unchanged', force = 0):
    build.transform.Transform.__init__(self)
    self.sourceDB      = sourceDB
    self.inputTag      = inputTag
    if isinstance(self.inputTag, str): self.inputTag = [self.inputTag]
    self.force         = force
    self.changed       = build.fileset.FileSet(tag = changedTag)
    self.unchanged     = build.fileset.FileSet(tag = unchangedTag)
    self.output.children.append(self.changed)
    self.output.children.append(self.unchanged)
    return

  def compare(self, source, sourceEntry):
    '''Return True if the checksum for "source" has changed since "sourceEntry" was recorded'''
    self.debugPrint('Checking for '+source+' in the source database', 3, 'sourceDB')
    checksum = self.sourceDB.getChecksum(source)
    if not sourceEntry[0] == checksum:
      self.debugPrint(source+' has changed relative to the source database: '+str(sourceEntry[0])+' <> '+str(checksum), 3, 'sourceDB')
      return 1
    return 0

  def hasChanged(self, source):
    '''Returns True if "source" has changed since it was last updates in the source database'''
    if self.force:
      self.debugPrint(source+' was forcibly tagged', 3, 'sourceDB')
      return 1
    try:
      if not os.path.exists(source):
        self.debugPrint(source+' does not exist', 3, 'sourceDB')
      else:
        if not self.compare(source, self.sourceDB[source]):
          for dep in self.sourceDB[source][3]:
            try:
              if self.compare(dep, self.sourceDB[dep]):
                return 1
            except KeyError: pass
          return 0
    except KeyError:
      self.debugPrint(source+' does not exist in source database', 3, 'sourceDB')
    return 1

  def handleFile(self, f, set):
    '''Place the file into either the "changed" or "unchanged" output set
       - If inputTag was specified, only handle files with this tag'''
    if self.inputTag is None or set.tag in self.inputTag:
      if self.hasChanged(f):
        self.changed.append(f)
      else:
        self.unchanged.append(f)
      return self.output
    return build.transform.Transform.handleFile(self, f, set)

class GenericTag (FileChanged):
  '''Uses input tag, extension and directory checks to group files which need further processing'''
  def __init__(self, sourceDB, outputTag, inputTag = None, ext = '', deferredExt = None, root = None, force = 0):
    FileChanged.__init__(self, sourceDB, inputTag, outputTag, 'old '+outputTag, force)
    self.ext   = ext
    if isinstance(self.ext, list):
      self.ext = map(lambda x: '.'+x, self.ext)
    elif isinstance(self.ext, str):
      self.ext = ['.'+self.ext]
    self.deferredExt   = deferredExt
    if isinstance(self.deferredExt, list):
      self.deferredExt = map(lambda x: '.'+x, self.deferredExt)
    elif isinstance(self.deferredExt, str):
      self.deferredExt = ['.'+self.deferredExt]
    self.root   = root
    if not self.root is None:
      self.root = os.path.normpath(self.root)
    self.deferredUpdates = build.fileset.FileSet(tag = 'update '+outputTag)
    self.output.children.append(self.deferredUpdates)
    return

  def __str__(self):
    return 'Tag transform for extension '+str(self.ext)+str(self.inputTag)+' to tag '+self.changed.tag

  def handleFile(self, f, set):
    '''- If the file is not in the specified root directory, use the default handler
       - If the file is in the extension list, call the parent method
       - If the file is in the deferred extension list and has changed, put it in the update set'''
    if self.inputTag is None or set.tag in self.inputTag:
      (base, ext) = os.path.splitext(f)
      if not self.root or self.root+os.sep == os.path.commonprefix([os.path.normpath(base), self.root+os.sep]):
        if self.ext is None or ext in self.ext:
          return FileChanged.handleFile(self, f, set)
        elif not self.deferredExt is None and ext in self.deferredExt:
          if self.hasChanged(f):
            self.deferredUpdates.append(f)
          return self.output
    return build.transform.Transform.handleFile(self, f, set)

  def handleFileSet(self, set):
    '''Check root directory if given, and then execute the default set handling method'''
    if self.root and not os.path.isdir(self.root):
      raise RuntimeError('Invalid root directory for tagging operation: '+self.root)
    return FileChanged.handleFileSet(self, set)

class Update (build.transform.Transform):
  '''Update nodes process files whose update in the source database was delayed'''
  def __init__(self, sourceDB, tag = None):
    build.transform.Transform.__init__(self)
    self.sourceDB = sourceDB
    if tag is None:
      self.tag = []
    else:
      self.tag = tag
    if self.tag and not isinstance(self.tag, list):
      self.tag = [self.tag]
    self.tag   = map(lambda t: 'update '+t, self.tag)
    return

  def __str__(self):
    return 'Update transform for '+str(self.tag)

  def handleFile(self, f, set):
    '''If the file tag starts with "update", then update it in the source database'''
    if (self.tag and set.tag in self.tag) or (set.tag and set.tag[:6] == 'update'):
      if os.path.isfile(f):
        self.sourceDB.updateSource(f)
      return self.output
    return build.transform.Transform.handleFile(self, f, set)

  def handleFileSet(self, set):
    '''Execute the default set handling method, and save source database'''
    output = build.transform.Transform.handleFileSet(self, set)
    # I could check here, and only save in the first recursive call
    self.sourceDB.save()
    try:
      import gc
      gc.collect()
    except ImportError: pass
    return output
