.. _ch_buildsystem:

BuildSystem
-----------

``BuildSystem`` is used for configuring PETSc before it gets compiled.
It is much like `GNU Autoconf (configure) <https://www.gnu.org/savannah-checkouts/gnu/autoconf/manual/autoconf-2.71/html_node/index.html#Top>`__
but written in Python especially for PETSc.


What is a build?
~~~~~~~~~~~~~~~~

The build stage compiles source to object files, stores them somehow
(usually in archives), and links shared libraries and executables. These
are mechanical operations that reduce to applying a construction rule to
sets of files. The `Make <http://www.gnu.org/software/make/>`__ tool is
great at this job. However, other parts of Make are not as useful, and
we should distinguish the two.

Make uses a single predicate, "older than", to decide whether to apply a
rule. This is a disaster. A useful upgrade to make would expand the list
of available predicates, including things like "md5sum has changed" and
"flags have changed". There have been attempts to use Make to determine
whether a file has changed, for example by using stamp files. However,
it cannot be done without severe contortions which make it much harder
to see what Make is doing and maintain the system. Right now, we can
combine make with the `ccache <https://ccache.samba.org/>`__ utility to
minimize recompiling and relinking.

Why is configure necessary?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``configure`` program is designed to assemble all information and preconditions
necessary for the build stage. This is a far more complicated task, heavily dependent on
the local hardware and software environment. It is also the source of nearly every build
problem. The most crucial aspect of a configure system is not performance, scalability, or
even functionality, but *debuggability*. Configuration failure is at least as common as
success, due to broken tools, operating system upgrades, hardware incompatibilities, user
error, and a host of other reasons. Problem diagnosis is the single biggest bottleneck for
development and maintenance time. Unfortunately, current systems are built to optimize the
successful case rather than the unsuccessful. In PETSc, we have developed the
``BuildSystem`` package to remedy the shortcomings of configuration systems such as
Autoconf, CMake, and SCons.

Why use PETSc BuildSystem?
~~~~~~~~~~~~~~~~~~~~~~~~~~

As several configuration tools
currently exist, it is instructive to consider why PETSc would choose to create another
from scratch. Below we list features and design considerations which lead us to prefer
``BuildSystem`` to the alternatives.

Namespacing
^^^^^^^^^^^

``BuildSystem`` wraps collections of related tests in Python modules, which also hold
the test results. Thus results are accessed using normal Python
namespacing. As rudimentary as this sounds, no namespacing beyond the
use of variable name prefixes is present in Autoconf, CMake, and SCons.
Instead, a flat namespace is used, mirroring the situation in C. This
tendency appears again when composing command lines for external tools,
such as the compiler and linker. In the traditional configure tools,
options are aggregated in a single bucket variable, such as ``INCLUDE``
or ``LIBS``, whereas in ``BuildSystem`` one can trace the provenance of a flag before it
is added to the command line. CMake also makes the unfortunate decision
to force all link options to resolve to full paths, which causes havoc
with compiler-private libraries.

Explicit control flow
^^^^^^^^^^^^^^^^^^^^^

The ``BuildSystem`` configure modules mentioned above, containing one ``Configure`` object
per module, are organized explicitly into a directed acyclic graph
(DAG). The user indicates dependence, an *edge* in the dependence graph,
with a single call, ``requires('path.to.other.test', self)``, which not
only structures the DAG, but returns the ``Configure`` object. The caller
can then use this object to access the results of the tests run by the
dependency, achieving test and result encapsulation simply.

Multi-language tests
^^^^^^^^^^^^^^^^^^^^

``BuildSystem`` maintains an explicit language stack, so that the current language
can be manipulated by the test environment. A compile or link can be run
using any language, complete with the proper compilers, flags,
libraries, etc., with a single call. This automation is crucial
for cross-language tests, which are thinly supported in current
tools. In fact, the design of these tools inhibits this kind of check.
The ``check_function_exists()`` call in Autoconf and CMake looks only
for the presence of a particular symbol in a library, and fails in C++
and on Microsoft Windows, whereas the equivalent ``BuildSystem`` test can also take a
declaration. The ``try_compile()`` test in Autoconf and CMake requires
the entire list of libraries be present in the ``LIBS`` variable,
providing no good way to obtain libraries from other tests in a modular
fashion. As another example, if the user has a dependent library that
requires ``libstdc++``, but they are working with a C project, no
straightforward method exists to add this dependency.

Subpackages
^^^^^^^^^^^

The most complicated, yet perhaps most useful, part of ``BuildSystem`` is
support for dependent packages. It provides an object scaffolding for
including a 3rd party package (more than 100 are now available) so that
PETSc downloads, builds, and tests the package for inclusion. The native
configure and build system for the package is used, and special support
exists for Autoconf and CMake packages. No similar system exists in the other
tools, which rely on static declarations, such as ``pkg-config`` or
``FindPackage.cmake`` files, that are not tested and often become
obsolete. They also require that any dependent packages use the same
configuration and build system.

Batch environments
^^^^^^^^^^^^^^^^^^

Most systems, such as Autoconf and CMake, do not actually run tests in a
batch environment, but rather require a direct specification, in CMake a
"platform file". This requires a human expert to write and maintain the
platform file. Alternatively, ``BuildSystem`` submits a dynamically
generated set of tests to the batch system, enabling automatic
cross-configuration and cross-compilation.

Caching
^^^^^^^

Caching often seems like an attractive option since configuration can be
quite time-consuming, and both Autoconf and CMake enable caching by
default. However, no system has the ability to reliably invalidate the
cache when the environment for the configuration changes. For example, a
compiler or library dependency may be upgraded on the system. Moreover,
dependencies between cached variables are not tracked, so that even if
some variables are correctly updated after an upgrade, others which
depend on them may not be. Moreover, CMake mixes together information
which is discovered automatically with that explicitly provided by the
user, which is often not tested.

Concision
^^^^^^^^^

The cognitive load is usually larger for larger code bases,
and our observation is that the addition of logic to Autoconf
and CMake is often quite cumbersome and verbose as they do not employ a modern,
higher level language. Although ``BuildSystem`` itself is not widely used,
it has the advantage of being written in a widely-understood, high-level
language.


High level organization
~~~~~~~~~~~~~~~~~~~~~~~

A minimal ``BuildSystem`` setup consists of a ``config`` directory off the
package root, which contains all the Python necessary to run (in addition
to the ``BuildSystem`` source). At minimum, the ``config`` directory contains
``configure.py``, which is executed to run the configure process, and a
module for the package itself. For example, PETSc contains
``config/PETSc/petsc.py``. It is also common to include a top level
``configure`` file to execute the configure, as this looks like
Autotools,

.. code-block:: python

   #!/usr/bin/env python
   import os
   execfile(os.path.join(os.path.dirname(__file__), 'config', 'configure.py'))

The ``configure.py`` script constructs a tree of configure modules and
executes the configure process over it. A minimal version of this would
be

.. code-block:: python

   package = 'PETSc'

   def configure(configure_options):
     # Command line arguments take precedence (but don't destroy argv[0])
     sys.argv = sys.argv[:1] + configure_options + sys.argv[1:]
     framework = config.framework.Framework(['--configModules='+package+'.Configure', '--optionsModule='+package+'.compilerOptions']+sys.argv[1:], loadArgDB = 0)
     framework.setup()
     framework.configure(out = sys.stdout)
     framework.storeSubstitutions(framework.argDB)
     framework.printSummary()
     framework.argDB.save(force = True)
     framework.logClear()
     framework.closeLog()

   if __name__ == '__main__':
     configure([])

The PETSc ``configure.py`` is quite a bit longer than this, as it
performs specialized command line processing, error handling, and
integrating logging with the rest of PETSc.

The ``config/package/Configure.py`` module determines how the tree of
``Configure`` objects is built and how the configure information is output.
The ``configure()`` method of the module will be run by the ``Framework``
object created at the top level. A minimal ``configure()`` method would look
like

.. code-block:: python

   def configure(self):
     self.framework.header          = self.arch.arch+'/include/'+self.project+'conf.h'
     self.framework.makeMacroHeader = self.arch.arch+'/conf/'+self.project+'variables'
     self.framework.makeRuleHeader  = self.arch.arch+'/conf/'+self.project+'rules'

     self.Dump()
     self.logClear()
     return

The ``Dump`` method runs over the tree of configure modules, and outputs
the data necessary for building, usually employing the
``addMakeMacro()``, ``addMakeRule()`` and ``addDefine()`` methods. These
methods funnel output to the include and make files defined by the
framework object, and set at the beginning of this ``configure()``
method. There is also some simple information that is often used, which
we define in the initializer,

.. code-block:: python

   def __init__(self, framework):
     config.base.Configure.__init__(self, framework)
     self.Project      = 'PETSc'
     self.project      = self.Project.lower()
     self.PROJECT      = self.Project.upper()
     self.headerPrefix = self.PROJECT
     self.substPrefix  = self.PROJECT
     self.framework.Project = self.Project
     return

More sophisticated configure assemblies, like PETSc, output some other
custom information, such as information about the machine, configure
process, and a script to recreate the configure run.

The `Package` configure module has two other main functions. First, top
level options can be defined in the ``setupHelp()`` method,

.. code-block:: python

   def setupHelp(self, help):
     import nargs
     help.addArgument(self.Project, '-prefix=<path>', nargs.Arg(None, '', 'Specify location to install '+self.Project+' (eg. /usr/local)'))
     help.addArgument(self.Project, '-load-path=<path>', nargs.Arg(None, os.path.join(os.getcwd(), 'modules'), 'Specify location of auxiliary modules'))
     help.addArgument(self.Project, '-with-shared-libraries', nargs.ArgBool(None, 0, 'Make libraries shared'))
     help.addArgument(self.Project, '-with-dynamic-loading', nargs.ArgBool(None, 0, 'Make libraries dynamic'))
     return

This uses the ``BuildSystem`` help facility that is used to define options
for all configure modules. The first argument groups these options into
a section named for the package. The second task is to build the tree of
modules for the configure run, using the ``setupDependencies()`` method.
A simple way to do this is by explicitly declaring dependencies,

.. code-block:: python

   def setupDependencies(self, framework):
       config.base.Configure.setupDependencies(self, framework)
       self.setCompilers  = framework.require('config.setCompilers',                self)
       self.arch          = framework.require(self.Project+'.utilities.arch',       self.setCompilers)
       self.projectdir    = framework.require(self.Project+'.utilities.projectdir', self.arch)
       self.compilers     = framework.require('config.compilers',                   self)
       self.types         = framework.require('config.types',                       self)
       self.headers       = framework.require('config.headers',                     self)
       self.functions     = framework.require('config.functions',                   self)
       self.libraries     = framework.require('config.libraries',                   self)

       self.compilers.headerPrefix  = self.headerPrefix
       self.types.headerPrefix      = self.headerPrefix
       self.headers.headerPrefix    = self.headerPrefix
       self.functions.headerPrefix  = self.headerPrefix
       self.libraries.headerPrefix  = self.headerPrefix

The ``projectdir`` and ``arch`` modules define the project root
directory and a build name so that multiple independent builds can be
managed. The ``Framework.require()`` method creates an edge in the
dependency graph for configure modules, and returns the module object so
that it can be queried after the configure information is determined.
Setting the header prefix routes all the defines made inside those
modules to our package configure header. We can also automatically
create configure modules based upon what we see on the filesystem,

.. code-block:: python

   for utility in os.listdir(os.path.join('config', self.Project, 'utilities')):
     (utilityName, ext) = os.path.splitext(utility)
     if not utilityName.startswith('.') and not utilityName.startswith('#') and ext == '.py' and not utilityName == '__init__':
       utilityObj                    = self.framework.require(self.Project+'.utilities.'+utilityName, self)
       utilityObj.headerPrefix       = self.headerPrefix
       utilityObj.archProvider       = self.arch
       utilityObj.languageProvider   = self.languages
       utilityObj.precisionProvider  = self.scalartypes
       utilityObj.installDirProvider = self.installdir
       utilityObj.externalPackagesDirProvider = self.externalpackagesdir
       setattr(self, utilityName.lower(), utilityObj)

The provider modules customize the information given to the module based
upon settings for our package. For example, PETSc can be compiled with a
scalar type that is single, double, or quad precision, and thus has a
``precisionProvider``. If a package does not have this capability, the
provider setting can be omitted.

Main objects
~~~~~~~~~~~~

Framework
^^^^^^^^^

The ``config.framework.Framework`` object serves as the central control
for a configure run. It maintains a graph of all the configure modules
involved, which is also used to track dependencies between them. It
initiates the run, compiles the results, and handles the final output.
It maintains the help list for all options available in the run. The
``setup()`` method performs generic ``Script`` setup and then is called
recursively on all the child modules. The ``cleanup()`` method performs
the final output and logging actions,

-  Substitute files

-  Output configure header

-  Log filesystem actions

Children may be added to the Framework using ``addChild()`` or
``getChild()``, but the far more frequent method is to use
``require()``. Here a module is requested, as in ``getChild()``, but it
is also required to run before another module, usually the one executing
the ``require()``. This provides a simple local interface to establish
dependencies between the child modules, and provides a partial order on
the children to the Framework.

A backwards compatibility mode is provided for which the user specifies
a configure header and set of files to experience substitution,
mirroring the common usage of Autoconf. Slight improvements have been
made in that all defines are now guarded, various prefixes are allowed
for defines and substitutions, and C specific constructs such as
function prototypes and typedefs are removed to a separate header.
However, this is not the intended future usage. The use of configure
modules by other modules in the same run provides a model for the
suggested interaction of a new build system with the Framework. If a
module requires another, it merely executes a ``require()``. For
instance, the PETSc configure module for HYPRE requires information
about MPI, and thus contains

.. code-block:: python

       self.mpi = self.framework.require("config.packages.MPI", self)

Notice that passing self for the last arguments means that the MPI
module will run before the HYPRE module. Furthermore, we save the
resulting object as ``self.mpi`` so that we may interrogate it later.
HYPRE can initially test whether MPI was indeed found using
``self.mpi.found``. When HYPRE requires the list of MPI libraries in
order to link a test object, the module can use ``self.mpi.lib``.

Base
^^^^

The ``config.base.Configure`` is the base class for all configure
objects. It handles several types of interaction. First, it has hooks
that allow the Framework to initialize it correctly. The Framework will
first instantiate the object and call ``setupDependencies()``. All
``require()`` calls should be made in that method. The Framework will
then call ``configure()``. If it succeeds, the object will be marked as
configured. Second, all configure tests should be run using
``executeTest()`` which formats the output and adds metadata for the
log.

Third, all tests that involve preprocessing, compiling, linking, and
running operator through ``base``. Two forms of this check are provided
for each operation. The first is an "output" form which is intended to
provide the status and complete output of the command. The second, or
"check" form will return a success or failure indication based upon the
status and output. The routines are

.. code-block:: python

     outputPreprocess(), checkPreprocess(), preprocess()
     outputCompile(),    checkCompile()
     outputLink(),       checkLink()
     outputRun(),        checkRun()

The language used for these operation is managed with a stack, similar
to Autoconf, using ``pushLanguage()`` and ``popLanguage()``. We also
provide special forms used to check for valid compiler and linker flags,
optionally adding them to the defaults.

.. code-block:: python

     checkCompilerFlag(), addCompilerFlag()
     checkLinkerFlag(),   addLinkerFlag()

You can also use ``getExecutable()`` to search for executables.

After configure tests have been run, various kinds of output can be
generated.A #define statement can be added to the configure header using
``addDefine()``, and ``addTypedef()`` and ``addPrototype()`` also put
information in this header file. Using ``addMakeMacro()`` and
``addMakeRule()`` will add make macros and rules to the output makefiles
specified in the framework. In addition we provide ``addSubstitution()``
and ``addArgumentSubstitution()`` to mimic the behavior of Autoconf if
necessary. The object may define a ``headerPrefix`` member, which will
be appended, followed by an underscore, to every define which is output
from it. Similarly, a ``substPrefix`` can be defined which applies to
every substitution from the object. Typedefs and function prototypes are
placed in a separate header in order to accommodate languages such as
Fortran whose preprocessor can sometimes fail at these statements.
