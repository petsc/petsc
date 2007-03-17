import RDict

import ihooks
import imp

class Hooks(ihooks.Hooks):
  def __init__(self):
    import os
    import sys

    ihooks.Hooks.__init__(self)
    return

  # sys interface replacement
  def default_path(self):
    import sys
    return sys.path

class Loader(ihooks.FancyModuleLoader):
  def find_module(self, name, path = None):
    if path is None:
      path = [None] + self.default_path()
    modules = []
    for dir in path:
      stuff = self.find_module_in_dir(name, dir)
      if stuff:
        if len(modules) > 0 and not stuff[2][2] == imp.PKG_DIRECTORY:
          continue
        modules.append(stuff)
    if modules: return modules
    return None

class Importer(ihooks.ModuleImporter):
  def import_module(self, name, globals = None, locals = None, fromlist = None):
    parent  = self.determine_parent(globals)
    q, tail = self.find_head_package(parent, name)
    mod     = self.load_tail(q, tail)
    if not fromlist:
      return q
    if hasattr(mod, "__path__"):
      self.ensure_fromlist(mod, fromlist)
    return mod

  def determine_parent(self, globals):
    if not globals or not globals.has_key("__name__"):
      return None
    pname = globals['__name__']
    if globals.has_key("__path__"):
      parent = self.modules[pname]
      assert globals is parent.__dict__
      return parent
    if '.' in pname:
      i = pname.rfind('.')
      pname  = pname[:i]
      parent = self.modules[pname]
      assert parent.__name__ == pname
      return parent
    return None

  def find_head_package(self, parent, name):
    if '.' in name:
      i = name.find('.')
      head = name[:i]
      tail = name[i+1:]
    else:
      head = name
      tail = ""
    if parent:
      qname = "%s.%s" % (parent.__name__, head)
    else:
      qname = head
    q = self.import_it(head, qname, parent)
    if q: return q, tail
    if parent:
      qname = head
      parent = None
      q = self.import_it(head, qname, parent)
      if q: return q, tail
    raise ImportError, "No module named " + qname

  def load_tail(self, q, tail):
    m = q
    while tail:
      i = tail.find('.')
      if i < 0: i = len(tail)
      head, tail = tail[:i], tail[i+1:]
      mname = "%s.%s" % (m.__name__, head)
      m = self.import_it(head, mname, m)
      if not m:
        raise ImportError, "No module named " + mname
    return m

  def ensure_fromlist(self, m, fromlist, recursive=0):
    for sub in fromlist:
      if sub == "*":
        if not recursive:
          try:
            all = m.__all__
          except AttributeError:
            pass
          else:
            self.ensure_fromlist(m, all, 1)
        continue
      if sub != "*" and not hasattr(m, sub):
        subname = "%s.%s" % (m.__name__, sub)
        submod = self.import_it(sub, subname, m)
        if not submod:
          raise ImportError, 'No symbol '+sub+' in module '+m.__name__

  def import_it(self, partname, fqname, parent, force_load=0):
    if not partname:
      raise ValueError, "Empty module name"
    if not force_load:
      try:
        return self.modules[fqname]
      except KeyError:
        pass
    try:
      path = parent and parent.__path__
    except AttributeError:
      return None
    stuff = self.loader.find_module(partname, path)
    if not stuff:
      return None
    path = []
    for s in stuff[1:]:
      m = self.loader.load_module(fqname, s)
      if hasattr(m, '__path__'):
        path.extend(m.__path__)
    mod = self.loader.load_module(fqname, stuff[0])
    if hasattr(mod, '__path__'):
      mod.__path__.extend(path)
    else:
      mod.__path__ = path
    if parent:
      setattr(parent, partname, mod)
    return mod

  def reload(self, module):
    name = module.__name__
    if '.' not in name:
      return self.import_it(name, name, None, force_load=1)
    i = name.rfind('.')
    pname = name[:i]
    parent = self.modules[pname]
    return self.import_it(name[i+1:], name, parent, force_load=1)

# Setup custom loading
loader   = Loader(Hooks())
importer = Importer(loader)
importer.install()
