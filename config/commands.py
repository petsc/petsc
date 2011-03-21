import builder

from mercurial.i18n import _
from mercurial.i18n import gettext
import mercurial.cmdutil as cmdutil
import mercurial.encoding as encoding
import mercurial.error as error
import mercurial.extensions as extensions
import mercurial.help as help
import mercurial.minirst as minirst
import mercurial.revset as revset
import mercurial.util as util

import os

def build(ui, repo, *pats, **opts):
  '''Compile all out of date source and generate a new shared library
  Any files specified will be added to the source database if not already present'''
  # TODO: Make rootDir an option (this should really be a filter for sourceDB)
  maker = builder.PETScMaker()
  maker.setup()
  maker.updateDependencies('libpetsc', maker.rootDir)
  for p in pats:
    filename = os.path.abspath(p)
    if not maker.sourceDatabase.hasNode(filename):
      maker.logPrint('Adding %s to the source database' % filename)
      maker.sourceDatabase.setNode(filename, [])
  if maker.buildLibraries('libpetsc', maker.rootDir):
    # This is overkill, but right now it is cheap
    maker.rebuildDependencies('libpetsc', maker.rootDir)
  maker.cleanup()
  return 0

def check(ui, repo, *pats, **opts):
  '''Check that build is functional'''
  import shutil

  maker = builder.PETScMaker()
  maker.setup()
  # C test
  if len(pats):
    examples = [os.path.abspath(p) for p in pats]
  else:
    examples = [os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5.c')]
  # Fortran test
  if hasattr(maker.configInfo.compilers, 'FC'):
    if maker.configInfo.fortrancpp.fortranDatatypes:
      examples.append(os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5f90t.F'))
    elif self.configInfo.compilers.fortranIsF90:
      examples.append(os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5f90.F'))
    else:
      examples.append(os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5f.F'))
  for ex in examples:
    exampleName = os.path.splitext(os.path.basename(ex))[0]
    exampleDir  = os.path.dirname(ex)
    objDir      = maker.getObjDir(exampleName)
    if os.path.isdir(objDir): shutil.rmtree(objDir)
    os.mkdir(objDir)
    objects     = maker.buildFile(ex, objDir)
    if not len(objects):
      print 'TEST FAILED (check make.log for details)'
      return 1
    executable = os.path.splitext(objects[0])[0]
    paramKey   = os.path.join(os.path.relpath(exampleDir, maker.petscDir), os.path.basename(executable))
    if paramKey in builder.regressionRequirements:
      if not builder.regressionRequirements[paramKey].issubset(packageNames):
        raise RuntimeError('This test requires packages: %s' % builder.regressionRequirements[paramKey])
    maker.link(executable, objects, maker.configInfo.languages.clanguage)
    if maker.runTest(exampleDir, executable, 1, **builder.regressionParameters.get(paramKey, {})):
      print 'TEST FAILED (check make.log for details)'
      return 1
    if os.path.isdir(objDir): shutil.rmtree(objDir)
  print 'All tests pass'
  maker.cleanup()
  return 0

def clean(ui, repo, *pats, **opts):
  '''Remove source database and all objects'''
  maker = builder.PETScMaker()
  maker.setup()
  maker.clean('libpetsc')
  maker.cleanup()
  return 0

def stubs(ui, repo, *pats, **opts):
  '''Compile all out of date source and generate a new shared library'''
  maker = builder.PETScMaker()
  maker.setup()
  for language in pats:
    getattr(maker, 'build'+language.capitalize()+'Stubs')()
  maker.cleanup()
  return 0

def help_(ui, name=None, with_version=False, unknowncmd=False):
    """show help for a given topic or a help overview

    With no arguments, print a list of commands with short help messages.

    Given a topic, extension, or command name, print help for that
    topic.

    Returns 0 if successful.
    """
    option_lists = []
    textwidth = ui.termwidth() - 2

    def addglobalopts(aliases):
        if ui.verbose:
            option_lists.append((_("global options:"), globalopts))
            if name == 'shortlist':
                option_lists.append((_('use "builder help" for the full list '
                                       'of commands'), ()))
        else:
            if name == 'shortlist':
                msg = _('use "builder help" for the full list of commands '
                        'or "builder -v" for details')
            elif aliases:
                msg = _('use "builder -v help%s" to show aliases and '
                        'global options') % (name and " " + name or "")
            else:
                msg = _('use "builder -v help %s" to show global options') % name
            option_lists.append((msg, ()))

    def helpcmd(name):
        if with_version:
            version_(ui)
            ui.write('\n')

        try:
            aliases, entry = cmdutil.findcmd(name, table, strict=unknowncmd)
        except error.AmbiguousCommand, inst:
            # py3k fix: except vars can't be used outside the scope of the
            # except block, nor can be used inside a lambda. python issue4617
            prefix = inst.args[0]
            select = lambda c: c.lstrip('^').startswith(prefix)
            helplist(_('list of commands:\n\n'), select)
            return

        # check if it's an invalid alias and display its error if it is
        if getattr(entry[0], 'badalias', False):
            if not unknowncmd:
                entry[0](ui)
            return

        # synopsis
        if len(entry) > 2:
            if entry[2].startswith('hg'):
                ui.write("%s\n" % entry[2])
            else:
                ui.write('hg %s %s\n' % (aliases[0], entry[2]))
        else:
            ui.write('hg %s\n' % aliases[0])

        # aliases
        if not ui.quiet and len(aliases) > 1:
            ui.write(_("\naliases: %s\n") % ', '.join(aliases[1:]))

        # description
        doc = gettext(entry[0].__doc__)
        if not doc:
            doc = _("(no help text available)")
        if hasattr(entry[0], 'definition'):  # aliased command
            if entry[0].definition.startswith('!'):  # shell alias
                doc = _('shell alias for::\n\n    %s') % entry[0].definition[1:]
            else:
                doc = _('alias for: hg %s\n\n%s') % (entry[0].definition, doc)
        if ui.quiet:
            doc = doc.splitlines()[0]
        keep = ui.verbose and ['verbose'] or []
        formatted, pruned = minirst.format(doc, textwidth, keep=keep)
        ui.write("\n%s\n" % formatted)
        if pruned:
            ui.write(_('\nuse "hg -v help %s" to show verbose help\n') % name)

        if not ui.quiet:
            # options
            if entry[1]:
                option_lists.append((_("options:\n"), entry[1]))

            addglobalopts(False)

    def helplist(header, select=None):
        h = {}
        cmds = {}
        for c, e in table.iteritems():
            f = c.split("|", 1)[0]
            if select and not select(f):
                continue
            if (not select and name != 'shortlist' and
                e[0].__module__ != __name__):
                continue
            if name == "shortlist" and not f.startswith("^"):
                continue
            f = f.lstrip("^")
            if not ui.debugflag and f.startswith("debug"):
                continue
            doc = e[0].__doc__
            if doc and 'DEPRECATED' in doc and not ui.verbose:
                continue
            doc = gettext(doc)
            if not doc:
                doc = _("(no help text available)")
            h[f] = doc.splitlines()[0].rstrip()
            cmds[f] = c.lstrip("^")

        if not h:
            ui.status(_('no commands defined\n'))
            return

        ui.status(header)
        fns = sorted(h)
        m = max(map(len, fns))
        for f in fns:
            if ui.verbose:
                commands = cmds[f].replace("|",", ")
                ui.write(" %s:\n      %s\n"%(commands, h[f]))
            else:
                ui.write('%s\n' % (util.wrap(h[f], textwidth,
                                             initindent=' %-*s   ' % (m, f),
                                             hangindent=' ' * (m + 4))))

        if not ui.quiet:
            addglobalopts(True)

    def helptopic(name):
        for names, header, doc in help.helptable:
            if name in names:
                break
        else:
            raise error.UnknownCommand(name)

        # description
        if not doc:
            doc = _("(no help text available)")
        if hasattr(doc, '__call__'):
            doc = doc()

        ui.write("%s\n\n" % header)
        ui.write("%s\n" % minirst.format(doc, textwidth, indent=4))

    def helpext(name):
        try:
            mod = extensions.find(name)
            doc = gettext(mod.__doc__) or _('no help text available')
        except KeyError:
            mod = None
            doc = extensions.disabledext(name)
            if not doc:
                raise error.UnknownCommand(name)

        if '\n' not in doc:
            head, tail = doc, ""
        else:
            head, tail = doc.split('\n', 1)
        ui.write(_('%s extension - %s\n\n') % (name.split('.')[-1], head))
        if tail:
            ui.write(minirst.format(tail, textwidth))
            ui.status('\n\n')

        if mod:
            try:
                ct = mod.cmdtable
            except AttributeError:
                ct = {}
            modcmds = set([c.split('|', 1)[0] for c in ct])
            helplist(_('list of commands:\n\n'), modcmds.__contains__)
        else:
            ui.write(_('use "hg help extensions" for information on enabling '
                       'extensions\n'))

    def helpextcmd(name):
        cmd, ext, mod = extensions.disabledcmd(name, ui.config('ui', 'strict'))
        doc = gettext(mod.__doc__).splitlines()[0]

        msg = help.listexts(_("'%s' is provided by the following "
                              "extension:") % cmd, {ext: doc}, len(ext),
                            indent=4)
        ui.write(minirst.format(msg, textwidth))
        ui.write('\n\n')
        ui.write(_('use "hg help extensions" for information on enabling '
                   'extensions\n'))

    help.addtopichook('revsets', revset.makedoc)

    if name and name != 'shortlist':
        i = None
        if unknowncmd:
            queries = (helpextcmd,)
        else:
            queries = (helptopic, helpcmd, helpext, helpextcmd)
        for f in queries:
            try:
                f(name)
                i = None
                break
            except error.UnknownCommand, inst:
                i = inst
        if i:
            raise i

    else:
        # program name
        if ui.verbose or with_version:
            version_(ui)
        else:
            ui.status(_("BuildSystem Configuration and Build\n"))
        ui.status('\n')

        # list of commands
        if name == "shortlist":
            header = _('basic commands:\n\n')
        else:
            header = _('list of commands:\n\n')

        helplist(header)
        if name != 'shortlist':
            exts, maxlength = extensions.enabled()
            text = help.listexts(_('enabled extensions:'), exts, maxlength)
            if text:
                ui.write("\n%s\n" % minirst.format(text, textwidth))

    # list all option lists
    opt_output = []
    multioccur = False
    for title, options in option_lists:
        opt_output.append(("\n%s" % title, None))
        for option in options:
            if len(option) == 5:
                shortopt, longopt, default, desc, optlabel = option
            else:
                shortopt, longopt, default, desc = option
                optlabel = _("VALUE") # default label

            if _("DEPRECATED") in desc and not ui.verbose:
                continue
            if isinstance(default, list):
                numqualifier = " %s [+]" % optlabel
                multioccur = True
            elif (default is not None) and not isinstance(default, bool):
                numqualifier = " %s" % optlabel
            else:
                numqualifier = ""
            opt_output.append(("%2s%s" %
                               (shortopt and "-%s" % shortopt,
                                longopt and " --%s%s" %
                                (longopt, numqualifier)),
                               "%s%s" % (desc,
                                         default
                                         and _(" (default: %s)") % default
                                         or "")))
    if multioccur:
        msg = _("\n[+] marked option can be specified multiple times")
        if ui.verbose and name != 'shortlist':
            opt_output.append((msg, None))
        else:
            opt_output.insert(-1, (msg, None))

    if not name:
        ui.write(_("\nadditional help topics:\n\n"))
        topics = []
        for names, header, doc in help.helptable:
            topics.append((sorted(names, key=len, reverse=True)[0], header))
        topics_len = max([len(s[0]) for s in topics])
        for t, desc in topics:
            ui.write(" %-*s  %s\n" % (topics_len, t, desc))

    if opt_output:
        colwidth = encoding.colwidth
        # normalize: (opt or message, desc or None, width of opt)
        entries = [desc and (opt, desc, colwidth(opt)) or (opt, None, 0)
                   for opt, desc in opt_output]
        hanging = max([e[2] for e in entries])
        for opt, desc, width in entries:
            if desc:
                initindent = ' %s%s  ' % (opt, ' ' * (hanging - width))
                hangindent = ' ' * (hanging + 3)
                ui.write('%s\n' % (util.wrap(desc, textwidth,
                                             initindent=initindent,
                                             hangindent=hangindent)))
            else:
                ui.write("%s\n" % opt)

def version_(ui):
    '''Output version and copyright information'''
    ui.write(_("BuildSystem Configuration and Build (version %s)\n") % util.version())
    ui.status(_(
        "(see http://www.mcs.anl.gov/petsc/petsc-as/documentation/copyright.html for more information)\n"
        "Permission to use, reproduce, prepare derivative works, and to redistribute to others this software\n"
        "and its documentation is hereby granted, provided that this notice is retained thereon and on all copies\n"
        "or modifications. UChicago Argonne, LLC and all other contributors make no representations as to the\n"
        "suitability and operability of this software for any purpose. It is provided \"as is\" without express or implied warranty.\n"))

globalopts = [('', 'config', [], _('set/override config option (use \'section.name=value\')'), _('CONFIG')),
              ('', 'cwd',    '', _('change working directory'), _('DIR')),
              ('', 'debug', None, _('enable debugging output')),
              ('', 'encoding', encoding.encoding, _('set the charset encoding'), _('ENCODE')),
              ('', 'encodingmode', encoding.encodingmode, _('set the charset encoding mode'), _('MODE')),
              ('h', 'help', None, _('display help and exit')),
              ('y', 'noninteractive', None, _('do not prompt, assume \'yes\' for any required answers')),
              ('', 'profile', None, _('print command execution profile')),
              ('q', 'quiet', None, _('suppress output')),
              ('R', 'repository', '', _('repository root directory or name of overlay bundle file'), _('REPO')),
              ('', 'time', None, _('time how long the command takes')),
              ('', 'traceback', None, _('always print a traceback on exception')),
              ('v', 'verbose', None, _('enable additional output')),
              ('', 'version', None, _('output version information and exit'))]

dryrunopts = [('n', 'dry-run', None, _('do not perform actions, just print output'))]

# Leading ^ puts command on the 'shortlist'
table = {'^build':  (build, dryrunopts, _('[FILE]')),
         '^check':  (check, [], ''),
         '^clean':  (clean, [], ''),
         'help':    (help_, [], _('[TOPIC]')),
         '^stubs':  (stubs, [], _('LANGUAGE...')),
         'version': (version_, [])}

norepo = ("help")
optionalrepo = ("")
