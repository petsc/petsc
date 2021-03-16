Other PETSc Features
--------------------

PETSc on a process subset
~~~~~~~~~~~~~~~~~~~~~~~~~

Users who wish to employ PETSc routines on only a subset of processes
within a larger parallel job, or who wish to use a “manager” process to
coordinate the work of “worker” PETSc processes, should specify an
alternative communicator for ``PETSC_COMM_WORLD`` by directly setting
its value, for example to an existing ``MPI_COMM_WORLD``,

::

   PETSC_COMM_WORLD=MPI_COMM_WORLD; /* To use an existing MPI_COMM_WORLD */

*before* calling ``PetscInitialize()``, but, obviously, after calling
``MPI_Init()``.

.. _sec_options:

Runtime Options
~~~~~~~~~~~~~~~

Allowing the user to modify parameters and options easily at runtime is
very desirable for many applications. PETSc provides a simple mechanism
to enable such customization. To print a list of available options for a
given program, simply specify the option ``-help`` at
runtime, e.g.,

::

   mpiexec -n 1 ./ex1 -help

Note that all runtime options correspond to particular PETSc routines
that can be explicitly called from within a program to set compile-time
defaults. For many applications it is natural to use a combination of
compile-time and runtime choices. For example, when solving a linear
system, one could explicitly specify use of the Krylov subspace
technique BiCGStab by calling

::

   KSPSetType(ksp,KSPBCGS);

One could then override this choice at runtime with the option

::

   -ksp_type tfqmr

to select the Transpose-Free QMR algorithm. (See
:any:`chapter_ksp` for details.)

The remainder of this section discusses details of runtime options.

.. _the-options-database-1:

The Options Database
^^^^^^^^^^^^^^^^^^^^

Each PETSc process maintains a database of option names and values
(stored as text strings). This database is generated with the command
``PetscInitialize()``, which is listed below in its C/C++ and Fortran
variants, respectively:

::

   PetscInitialize(int *argc,char ***args,const char *file,const char *help); /* C */

.. code-block:: fortran

   call PetscInitialize(character file,integer ierr) ! Fortran

The arguments ``argc`` and ``args`` (in the C/C++ version only) are the
addresses of usual command line arguments, while the ``file`` is a name
of a file that can contain additional options. By default this file is
called ``.petscrc`` in the user’s home directory. The user can also
specify options via the environmental variable ``PETSC_OPTIONS``. The
options are processed in the following order:

#. file

#. environmental variable

#. command line

Thus, the command line options supersede the environmental variable
options, which in turn supersede the options file.

The file format for specifying options is

.. code-block:: none

   -optionname possible_value
   -anotheroptionname possible_value
   ...

All of the option names must begin with a dash (-) and have no
intervening spaces. Note that the option values cannot have intervening
spaces either, and tab characters cannot be used between the option
names and values. The user can employ any naming convention. For
uniformity throughout PETSc, we employ the format
``-[prefix_]package_option`` (for instance, ``-ksp_type``,
``-mat_view ::info``, or ``-mg_levels_ksp_type``).

Users can specify an alias for any option name (to avoid typing the
sometimes lengthy default name) by adding an alias to the ``.petscrc``
file in the format

.. code-block:: none

   alias -newname -oldname

For example,

.. code-block:: none

   alias -kspt -ksp_type
   alias -sd -start_in_debugger

Comments can be placed in the ``.petscrc`` file by using ``#`` in the
first column of a line.

Options Prefixes
^^^^^^^^^^^^^^^^

Options prefixes allow specific objects to be controlled from the
options database. For instance, ``PCMG`` gives prefixes to its nested
``KSP`` objects; one may control the coarse grid solver by adding the
``mg_coarse`` prefix, for example ``-mg_coarse_ksp_type preonly``. One
may also use ``KSPSetOptionsPrefix()``,\ ``DMSetOptionsPrefix()`` ,
``SNESSetOptionsPrefix()``, ``TSSetOptionsPrefix()``, and similar
functions to assign custom prefixes, useful for applications with
multiple or nested solvers.

Adding options from a file
^^^^^^^^^^^^^^^^^^^^^^^^^^

PETSc can load additional options from a file using ``PetscOptionsInsertFile()``,
which can also be used from the command line, e.g. ``-options_file my_options.opts``.

One can also use YAML files this way (relying on ``PetscOptionsInsertFileYAML()``).
For example, the following file:

.. literalinclude:: /../../../src/sys/tests/ex47-options.yaml
  :language: yaml

corresponds to the following PETSc options:

.. literalinclude:: /../../../src/sys/tests/output/ex47_3_options.out
  :language: none
  :start-after: #
  :end-before: #End

With ``-options_file``, PETSc will parse the file as YAML if it ends in a standard
YAML or JSON [#json]_ extension or if one uses a ``:yaml`` postfix,
e.g. ``-options_file my_options.yaml`` or ``-options_file my_options.txt:yaml``

PETSc will also check the first line of the options file itself and
parse the file as YAML if it matches certain criteria, for example.


.. literalinclude:: /../../../src/sys/tests/ex47-yaml_tag
  :language: yaml

and

.. literalinclude:: /../../../src/sys/tests/ex47-yaml_doc
  :language: yaml

both correspond to options

.. literalinclude:: /../../../src/sys/tests/output/ex47_2_auto.out
  :language: none
  :start-after: #
  :end-before: #End


User-Defined PetscOptions
^^^^^^^^^^^^^^^^^^^^^^^^^

Any subroutine in a PETSc program can add entries to the database with
the command

::

   PetscOptionsSetValue(PetscOptions options,char *name,char *value);

though this is rarely done. To locate options in the database, one
should use the commands

::

   PetscOptionsHasName(PetscOptions options,char *pre,char *name,PetscBool *flg);
   PetscOptionsGetInt(PetscOptions options,char *pre,char *name,PetscInt *value,PetscBool *flg);
   PetscOptionsGetReal(PetscOptions options,char *pre,char *name,PetscReal *value,PetscBool *flg);
   PetscOptionsGetString(PetscOptions options,char *pre,char *name,char *value,int maxlen,PetscBool  *flg);
   PetscOptionsGetStringArray(PetscOptions options,char *pre,char *name,char **values,PetscInt *nmax,PetscBool *flg);
   PetscOptionsGetIntArray(PetscOptions options,char *pre,char *name,int *value,PetscInt *nmax,PetscBool *flg);
   PetscOptionsGetRealArray(PetscOptions options,char *pre,char *name,PetscReal *value, PetscInt *nmax,PetscBool *flg);

All of these routines set ``flg=PETSC_TRUE`` if the corresponding option
was found, ``flg=PETSC_FALSE`` if it was not found. The optional
argument ``pre`` indicates that the true name of the option is the given
name (with the dash “-” removed) prepended by the prefix ``pre``.
Usually ``pre`` should be set to ``NULL`` (or ``PETSC_NULL_CHARACTER``
for Fortran); its purpose is to allow someone to rename all the options
in a package without knowing the names of the individual options. For
example, when using block Jacobi preconditioning, the ``KSP`` and ``PC``
methods used on the individual blocks can be controlled via the options
``-sub_ksp_type`` and ``-sub_pc_type``.

Keeping Track of Options
^^^^^^^^^^^^^^^^^^^^^^^^

One useful means of keeping track of user-specified runtime options is
use of ``-options_view``, which prints to ``stdout`` during
``PetscFinalize()`` a table of all runtime options that the user has
specified. A related option is ``-options_left``, which prints the
options table and indicates any options that have *not* been requested
upon a call to ``PetscFinalize()``. This feature is useful to check
whether an option has been activated for a particular PETSc object (such
as a solver or matrix format), or whether an option name may have been
accidentally misspelled.

.. _sec_viewers:

Viewers: Looking at PETSc Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSc employs a consistent scheme for examining, printing, and saving
objects through commands of the form

::

   XXXView(XXX obj,PetscViewer viewer);

Here ``obj`` is any PETSc object of type ``XXX``, where ``XXX`` is
``Mat``, ``Vec``, ``SNES``, etc. There are several predefined viewers.

-  Passing in a zero (``0``) for the viewer causes the object to be
   printed to the screen; this is useful when viewing an object in a
   debugger but should be avoided in source code.

-  ``PETSC_VIEWER_STDOUT_SELF`` and ``PETSC_VIEWER_STDOUT_WORLD`` causes
   the object to be printed to the screen.

-  ``PETSC_VIEWER_DRAW_SELF`` ``PETSC_VIEWER_DRAW_WORLD`` causes the
   object to be drawn in a default X window.

-  Passing in a viewer obtained by ``PetscViewerDrawOpen()`` causes the
   object to be displayed graphically. See
   :any:`sec_graphics` for more on PETSc’s graphics support.

-  To save an object to a file in ASCII format, the user creates the
   viewer object with the command
   ``PetscViewerASCIIOpen(MPI_Comm comm, char* file, PetscViewer *viewer)``.
   This object is analogous to ``PETSC_VIEWER_STDOUT_SELF`` (for a
   communicator of ``MPI_COMM_SELF``) and ``PETSC_VIEWER_STDOUT_WORLD``
   (for a parallel communicator).

-  To save an object to a file in binary format, the user creates the
   viewer object with the command
   ``PetscViewerBinaryOpen(MPI_Comm comm,char* file,PetscViewerBinaryType type, PetscViewer *viewer)``.
   Details of binary I/O are discussed below.

-  Vector and matrix objects can be passed to a running MATLAB process
   with a viewer created by
   ``PetscViewerSocketOpen(MPI_Comm comm,char *machine,int port,PetscViewer *viewer)``.
   For more, see :any:`sec_matlabsocket`.

The user can control the format of ASCII printed objects with viewers
created by ``PetscViewerASCIIOpen()`` by calling

::

   PetscViewerPushFormat(PetscViewer viewer,PetscViewerFormat format);

Formats include ``PETSC_VIEWER_DEFAULT``, ``PETSC_VIEWER_ASCII_MATLAB``,
and ``PETSC_VIEWER_ASCII_IMPL``. The implementation-specific format,
``PETSC_VIEWER_ASCII_IMPL``, displays the object in the most natural way
for a particular implementation.

The routines

::

   PetscViewerPushFormat(PetscViewer viewer,PetscViewerFormat format);
   PetscViewerPopFormat(PetscViewer viewer);

allow one to temporarily change the format of a viewer.

As discussed above, one can output PETSc objects in binary format by
first opening a binary viewer with ``PetscViewerBinaryOpen()`` and then
using ``MatView()``, ``VecView()``, etc. The corresponding routines for
input of a binary object have the form ``XXXLoad()``. In particular,
matrix and vector binary input is handled by the following routines:

::

   MatLoad(PetscViewer viewer,MatType outtype,Mat *newmat);
   VecLoad(PetscViewer viewer,VecType outtype,Vec *newvec);

These routines generate parallel matrices and vectors if the viewer’s
communicator has more than one process. The particular matrix and vector
formats are determined from the options database; see the manual pages
for details.

One can provide additional information about matrix data for matrices
stored on disk by providing an optional file ``matrixfilename.info``,
where ``matrixfilename`` is the name of the file containing the matrix.
The format of the optional file is the same as the ``.petscrc`` file and
can (currently) contain the following:

.. code-block:: none

   -matload_block_size <bs>

The block size indicates the size of blocks to use if the matrix is read
into a block oriented data structure (for example, ``MATMPIBAIJ``). The
diagonal information ``s1,s2,s3,...`` indicates which (block) diagonals
in the matrix have nonzero values.

.. _sec_viewfromoptions:

Viewing From Options
^^^^^^^^^^^^^^^^^^^^

Command-line options provide a particularly convenient way to view PETSc
objects. All options of the form ``-xxx_view`` accept
colon(``:``)-separated compound arguments which specify a viewer type,
format, and/or destination (e.g. file name or socket) if appropriate.
For example, to quickly export a binary file containing a matrix, one
may use ``-mat_view binary:matrix.out``, or to output to a
MATLAB-compatible ASCII file, one may use
``-mat_view ascii:matrix.m:ascii_matlab``. See the
``PetscOptionsGetViewer()`` man page for full details, as well as the
``XXXViewFromOptions()`` man pages (for instance,
``PetscDrawSetFromOptions()``) for many other convenient command-line
options.

Using Viewers to Check Load Imbalance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PetscViewer format ``PETSC_VIEWER_LOAD_BALANCE`` will cause certain
objects to display simple measures of their imbalance. For example

.. code-block:: none

   -n 4 ./ex32 -ksp_view_mat ::load_balance

will display

.. code-block:: none

     Nonzeros: Min 162  avg 168  max 174

indicating that one process has 162 nonzero entries in the matrix, the
average number of nonzeros per process is 168 and the maximum number of
nonzeros is 174. Similar for vectors one can see the load balancing
with, for example,

.. code-block:: none

   -n 4 ./ex32 -ksp_view_rhs ::load_balance

The measurements of load balancing can also be done within the program
with calls to the appropriate object viewer with the viewer format
``PETSC_VIEWER_LOAD_BALANCE``.

.. _sec_saws:

Using SAWs with PETSc
~~~~~~~~~~~~~~~~~~~~~

The Scientific Application Web server, SAWs [#saws]_, allows one to monitor
running PETSc applications from a browser. ``configure`` PETSc with
the additional option ``--download-saws``. Options to use SAWs include

-  ``-saws_options`` - allows setting values in the PETSc options
   database via the browser (works only on one process).

-  ``-stack_view saws`` - allows monitoring the current stack frame that
   PETSc is in; refresh to see the new location.

-  ``-snes_monitor_saws, -ksp_monitor_saws`` - monitor the solvers’
   iterations from the web browser.

For each of these you need to point your browser to
``http://hostname:8080``, for example ``http://localhost:8080``. Options
that control behavior of SAWs include

-  ``-saws_log filename`` - log all SAWs actions in a file.

-  ``-saws_https certfile`` - use HTTPS instead of HTTP with a
   certificate.

-  ``-saws_port_auto_select`` - have SAWs pick a port number instead of
   using 8080.

-  ``-saws_port port`` - use ``port`` instead of 8080.

-  ``-saws_root rootdirectory`` - local directory to which the SAWs
   browser will have read access.

-  ``-saws_local`` - use the local file system to obtain the SAWS
   javascript files (they much be in ``rootdirectory/js``).

Also see the manual pages for ``PetscSAWsBlock``,
``PetscObjectSAWsTakeAccess``, ``PetscObjectSAWsGrantAccess``,
``PetscObjectSAWsSetBlock``, ``PetscStackSAWsGrantAccess``
``PetscStackSAWsTakeAccess``, ``KSPMonitorSAWs``, and
``SNESMonitorSAWs``.

.. _sec-debugging:

Debugging
~~~~~~~~~

PETSc programs may be debugged using one of the two options below.

-  ``-start_in_debugger`` ``[noxterm,dbx,xxgdb,xdb,xldb,lldb]``
   ``[-display name]`` - start all processes in debugger

-  ``-on_error_attach_debugger`` ``[noxterm,dbx,xxgdb,xdb,xldb,lldb]``
   ``[-display name]`` - start debugger only on encountering an error

Note that, in general, debugging MPI programs cannot be done in the
usual manner of starting the programming in the debugger (because then
it cannot set up the MPI communication and remote processes).

By default the GNU debugger ``gdb`` is used when ``-start_in_debugger``
or ``-on_error_attach_debugger`` is specified. To employ either
``xxgdb`` or the common UNIX debugger ``dbx``, one uses command line
options as indicated above. On HP-UX machines the debugger ``xdb``
should be used instead of ``dbx``; on RS/6000 machines the ``xldb``
debugger is supported as well. On OS X systems with XCode tools,
``lldb`` is available. By default, the debugger will be started in a new
xterm (to enable running separate debuggers on each process), unless the
option ``noxterm`` is used. In order to handle the MPI startup phase,
the debugger command ``cont`` should be used to continue execution of
the program within the debugger. Rerunning the program through the
debugger requires terminating the first job and restarting the
processor(s); the usual ``run`` option in the debugger will not
correctly handle the MPI startup and should not be used. Not all
debuggers work on all machines, the user may have to experiment to find
one that works correctly.

You can select a subset of the processes to be debugged (the rest just
run without the debugger) with the option

.. code-block:: none

   -debugger_ranks rank1,rank2,...

where you simply list the ranks you want the debugger to run with.

.. _sec_errors:

Error Handling
~~~~~~~~~~~~~~

Errors are handled through the routine ``PetscError()``. This routine
checks a stack of error handlers and calls the one on the top. If the
stack is empty, it selects ``PetscTraceBackErrorHandler()``, which tries
to print a traceback. A new error handler can be put on the stack with

::

   PetscPushErrorHandler(PetscErrorCode (*HandlerFunction)(int line,char *dir,char *file,char *message,int number,void*),void *HandlerContext)

The arguments to ``HandlerFunction()`` are the line number where the
error occurred, the file in which the error was detected, the
corresponding directory, the error message, the error integer, and the
``HandlerContext.`` The routine

::

   PetscPopErrorHandler()

removes the last error handler and discards it.

PETSc provides two additional error handlers besides
``PetscTraceBackErrorHandler()``:

::

   PetscAbortErrorHandler()
   PetscAttachErrorHandler()

The function ``PetscAbortErrorHandler()`` calls abort on encountering an
error, while ``PetscAttachErrorHandler()`` attaches a debugger to the
running process if an error is detected. At runtime, these error
handlers can be set with the options ``-on_error_abort`` or
``-on_error_attach_debugger`` ``[noxterm, dbx, xxgdb, xldb]``
``[-display DISPLAY]``.

All PETSc calls can be traced (useful for determining where a program is
hanging without running in the debugger) with the option

.. code-block:: none

   -log_trace [filename]

where ``filename`` is optional. By default the traces are printed to the
screen. This can also be set with the command
``PetscLogTraceBegin(FILE*)``.

It is also possible to trap signals by using the command

::

   PetscPushSignalHandler( PetscErrorCode (*Handler)(int,void *),void *ctx);

The default handler ``PetscSignalHandlerDefault()`` calls
``PetscError()`` and then terminates. In general, a signal in PETSc
indicates a catastrophic failure. Any error handler that the user
provides should try to clean up only before exiting. By default all
PETSc programs use the default signal handler, although the user can
turn this off at runtime with the option ``-no_signal_handler`` .

There is a separate signal handler for floating-point exceptions. The
option ``-fp_trap`` turns on the floating-point trap at runtime, and the
routine

::

   PetscSetFPTrap(PetscFPTrap flag);

can be used in-line. A ``flag`` of ``PETSC_FP_TRAP_ON`` indicates that
floating-point exceptions should be trapped, while a value of
``PETSC_FP_TRAP_OFF`` (the default) indicates that they should be
ignored. Note that on certain machines, in particular the IBM RS/6000,
trapping is very expensive.

A small set of macros is used to make the error handling lightweight.
These macros are used throughout the PETSc libraries and can be employed
by the application programmer as well. When an error is first detected,
one should set it by calling

::

   SETERRQ(MPI_Comm comm,PetscErrorCode flag,,char *message);

The user should check the return codes for all PETSc routines (and
possibly user-defined routines as well) with

::

   ierr = PetscRoutine(...);CHKERRQ(PetscErrorCode ierr);

Likewise, all memory allocations should be checked with

::

   ierr = PetscMalloc1(n, &ptr);CHKERRQ(ierr);

If this procedure is followed throughout all of the user’s libraries and
codes, any error will by default generate a clean traceback of the
location of the error.

Note that the macro ``PETSC_FUNCTION_NAME`` is used to keep track of
routine names during error tracebacks. Users need not worry about this
macro in their application codes; however, users can take advantage of
this feature if desired by setting this macro before each user-defined
routine that may call ``SETERRQ()``, ``CHKERRQ()``. A simple example of
usage is given below.

::

   PetscErrorCode MyRoutine1() {
       /* Declarations Here */
       PetscFunctionBeginUser;
       /* code here */
       PetscFunctionReturn(0);
   }

.. _sec_complex:

Numbers
~~~~~~~

PETSc supports the use of complex numbers in application programs
written in C, C++, and Fortran. To do so, we employ either the C99
``complex`` type or the C++ versions of the PETSc libraries in which the
basic “scalar” datatype, given in PETSc codes by ``PetscScalar``, is
defined as ``complex`` (or ``complex<double>`` for machines using
templated complex class libraries). To work with complex numbers, the
user should run ``configure`` with the additional option
``--with-scalar-type=complex``. The
:doc:`installation instructions </install/index>`
provide detailed instructions for installing PETSc. You can use
``--with-clanguage=c`` (the default) to use the C99 complex numbers or
``--with-clanguage=c++`` to use the C++ complex type [#cxx_note]_.

Recall that each variant of the PETSc libraries is stored in a different
directory, given by ``$PETSC_DIR/lib/$PETSC_ARCH``

according to the architecture. Thus, the libraries for complex numbers
are maintained separately from those for real numbers. When using any of
the complex numbers versions of PETSc, *all* vector and matrix elements
are treated as complex, even if their imaginary components are zero. Of
course, one can elect to use only the real parts of the complex numbers
when using the complex versions of the PETSc libraries; however, when
working *only* with real numbers in a code, one should use a version of
PETSc for real numbers for best efficiency.

The program
`KSP Tutorial ex11 <../../src/ksp/ksp/tutorials/ex11.c.html>`__
solves a linear system with a complex coefficient matrix. Its Fortran
counterpart is
`KSP Tutorial ex11f <../../src/ksp/ksp/tutorials/ex11f.F90.html>`__.

Parallel Communication
~~~~~~~~~~~~~~~~~~~~~~

When used in a message-passing environment, all communication within
PETSc is done through MPI, the message-passing interface standard
:cite:`MPI-final`. Any file that includes ``petscsys.h`` (or
any other PETSc include file) can freely use any MPI routine.

.. _sec_graphics:

Graphics
~~~~~~~~

The PETSc graphics library is not intended to compete with high-quality
graphics packages. Instead, it is intended to be easy to use
interactively with PETSc programs. We urge users to generate their
publication-quality graphics using a professional graphics package. If a
user wants to hook certain packages into PETSc, he or she should send a
message to
`petsc-maint@mcs.anl.gov <mailto:petsc-maint@mcs.anl.gov>`__; we
will see whether it is reasonable to try to provide direct interfaces.

Windows as PetscViewers
^^^^^^^^^^^^^^^^^^^^^^^

For drawing predefined PETSc objects such as matrices and vectors, one
may first create a viewer using the command

::

   PetscViewerDrawOpen(MPI_Comm comm,char *display,char *title,int x,int y,int w,int h,PetscViewer *viewer);

This viewer may be passed to any of the ``XXXView()`` routines.
Alternately, one may use command-line options to quickly specify viewer
formats, including ``PetscDraw``-based ones; see
:any:`sec_viewfromoptions`.

To draw directly into the viewer, one must obtain the ``PetscDraw``
object with the command

::

   PetscViewerDrawGetDraw(PetscViewer viewer,PetscDraw *draw);

Then one can call any of the ``PetscDrawXXX`` commands on the ``draw``
object. If one obtains the ``draw`` object in this manner, one does not
call the ``PetscDrawOpenX()`` command discussed below.

Predefined viewers, ``PETSC_VIEWER_DRAW_WORLD`` and
``PETSC_VIEWER_DRAW_SELF``, may be used at any time. Their initial use
will cause the appropriate window to be created.

Implementations using OpenGL, TikZ, and other formats may be selected
with ``PetscDrawSetType()``. PETSc can also produce movies; see
``PetscDrawSetSaveMovie()``, and note that command-line options can also
be convenient; see the ``PetscDrawSetFromOptions()`` man page.

By default, PETSc drawing tools employ a private colormap, which
remedies the problem of poor color choices for contour plots due to an
external program’s mangling of the colormap. Unfortunately, this may
cause flashing of colors as the mouse is moved between the PETSc windows
and other windows. Alternatively, a shared colormap can be used via the
option ``-draw_x_shared_colormap``.

Simple PetscDrawing
^^^^^^^^^^^^^^^^^^^

With the default format, one can open a window that is not associated
with a viewer directly under the X11 Window System or OpenGL with the
command

::

   PetscDrawCreate(MPI_Comm comm,char *display,char *title,int x,int y,int w,int h,PetscDraw *win);
   PetscDrawSetFromOptions(win);

All drawing routines are performed relative to the window’s coordinate
system and viewport. By default, the drawing coordinates are from
``(0,0)`` to ``(1,1)``, where ``(0,0)`` indicates the lower left corner
of the window. The application program can change the window coordinates
with the command

::

   PetscDrawSetCoordinates(PetscDraw win,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr);

By default, graphics will be drawn in the entire window. To restrict the
drawing to a portion of the window, one may use the command

::

   PetscDrawSetViewPort(PetscDraw win,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr);

These arguments, which indicate the fraction of the window in which the
drawing should be done, must satisfy
:math:`0 \leq {\tt xl} \leq {\tt xr} \leq 1` and
:math:`0 \leq {\tt yl} \leq {\tt yr} \leq 1.`

To draw a line, one uses the command

::

   PetscDrawLine(PetscDraw win,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl);

The argument ``cl`` indicates the color (which is an integer between 0
and 255) of the line. A list of predefined colors may be found in
``include/petscdraw.h`` and includes ``PETSC_DRAW_BLACK``,
``PETSC_DRAW_RED``, ``PETSC_DRAW_BLUE`` etc.

To ensure that all graphics actually have been displayed, one should use
the command

::

   PetscDrawFlush(PetscDraw win);

When displaying by using double buffering, which is set with the command

::

   PetscDrawSetDoubleBuffer(PetscDraw win);

*all* processes must call

::

   PetscDrawFlush(PetscDraw win);

in order to swap the buffers. From the options database one may use
``-draw_pause`` ``n``, which causes the PETSc application to pause ``n``
seconds at each ``PetscDrawPause()``. A time of ``-1`` indicates that
the application should pause until receiving mouse input from the user.

Text can be drawn with commands

::

   PetscDrawString(PetscDraw win,PetscReal x,PetscReal y,int color,char *text);
   PetscDrawStringVertical(PetscDraw win,PetscReal x,PetscReal y,int color,const char *text);
   PetscDrawStringCentered(PetscDraw win,PetscReal x,PetscReal y,int color,const char *text);
   PetscDrawStringBoxed(PetscDraw draw,PetscReal sxl,PetscReal syl,int sc,int bc,const char text[],PetscReal *w,PetscReal *h);

The user can set the text font size or determine it with the commands

::

   PetscDrawStringSetSize(PetscDraw win,PetscReal width,PetscReal height);
   PetscDrawStringGetSize(PetscDraw win,PetscReal *width,PetscReal *height);

Line Graphs
^^^^^^^^^^^

PETSc includes a set of routines for manipulating simple two-dimensional
graphs. These routines, which begin with ``PetscDrawAxisDraw()``, are
usually not used directly by the application programmer. Instead, the
programmer employs the line graph routines to draw simple line graphs.
As shown in the :ref:`listing below <listing_draw_test_ex3>`, line
graphs are created with the command

::

   PetscDrawLGCreate(PetscDraw win,PetscInt ncurves,PetscDrawLG *ctx);

The argument ``ncurves`` indicates how many curves are to be drawn.
Points can be added to each of the curves with the command

::

   PetscDrawLGAddPoint(PetscDrawLG ctx,PetscReal *x,PetscReal *y);

The arguments ``x`` and ``y`` are arrays containing the next point value
for each curve. Several points for each curve may be added with

::

   PetscDrawLGAddPoints(PetscDrawLG ctx,PetscInt n,PetscReal **x,PetscReal **y);

The line graph is drawn (or redrawn) with the command

::

   PetscDrawLGDraw(PetscDrawLG ctx);

A line graph that is no longer needed can be destroyed with the command

::

   PetscDrawLGDestroy(PetscDrawLG *ctx);

To plot new curves, one can reset a linegraph with the command

::

   PetscDrawLGReset(PetscDrawLG ctx);

The line graph automatically determines the range of values to display
on the two axes. The user can change these defaults with the command

::

   PetscDrawLGSetLimits(PetscDrawLG ctx,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax);

It is also possible to change the display of the axes and to label them.
This procedure is done by first obtaining the axes context with the
command

::

   PetscDrawLGGetAxis(PetscDrawLG ctx,PetscDrawAxis *axis);

One can set the axes’ colors and labels, respectively, by using the
commands

::

   PetscDrawAxisSetColors(PetscDrawAxis axis,int axis_lines,int ticks,int text);
   PetscDrawAxisSetLabels(PetscDrawAxis axis,char *top,char *x,char *y);

It is possible to turn off all graphics with the option ``-nox``. This
will prevent any windows from being opened or any drawing actions to be
done. This is useful for running large jobs when the graphics overhead
is too large, or for timing.

The full example, `Draw Test ex3 <../../src/sys/classes/draw/tests/ex3.c.html>`__,
follows.

.. _listing_draw_test_ex3:

.. admonition:: Listing: ``src/classes/draw/tests/ex3.c``
   :name: snes-ex1

   .. literalinclude:: ../../../sys/classes/draw/tests/ex3.c
      :end-before: /*TEST



Graphical Convergence Monitor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For both the linear and nonlinear solvers default routines allow one to
graphically monitor convergence of the iterative method. These are
accessed via the command line with ``-ksp_monitor draw::draw_lg`` and
``-snes_monitor draw::draw_lg``. See also
:any:`sec_kspmonitor` and :any:`sec_snesmonitor`.

Disabling Graphics at Compile Time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To disable all X-window-based graphics, run ``configure`` with the
additional option ``--with-x=0``

.. _sec-emacs:

Emacs Users
~~~~~~~~~~~

Many PETSc developers use Emacs, which can be used as a "simple" text editor or a comprehensive development environment.
For a more integrated development environment, we recommend using `lsp-mode <https://emacs-lsp.github.io/lsp-mode/>`_ (or `eglot <https://github.com/joaotavora/eglot>`_) with `clangd <https://clangd.llvm.org/>`_.
The most convenient way to teach clangd what compilation flags to use is to install `Bear <https://github.com/rizsotto/Bear>`_ ("build ear") and run::

  bear make -B

which will do a complete rebuild (``-B``) of PETSc and capture the compilation commands in a file named ``compile_commands.json``, which will be automatically picked up by clangd.
You can use the same procedure when building examples or your own project.
It can also be used with any other editor that supports clangd, including VS Code and Vim.
When lsp-mode is accompanied by `flycheck <https://www.flycheck.org/en/latest/>`_, Emacs will provide real-time feedback and syntax checking, along with refactoring tools provided by clangd.

The easiest way to install packages in recent Emacs is to use the "Options" menu to select "Manage Emacs Packages".

Tags
^^^^

It is sometimes useful to cross-reference tags across projects.
Regardless of whether you use lsp-mode, it can be useful to use `GNU Global <https://www.gnu.org/software/global/>`_ (install ``gtags``) to provide reverse lookups (e.g. find all call sites
for a given function) across all projects you might work on/browse.
Tags for PETSc can be generated by running ``make allgtags`` from ``$PETSC_DIR``, or one can generate tags for all projects by running a command such as

.. code-block:: none

   find $PETSC_DIR/{include,src,tutorials,$PETSC_ARCH/include} any/other/paths \
      -regex '.*\.\(cc\|hh\|cpp\|cxx\|C\|hpp\|c\|h\|cu\)$' \
      | grep -v ftn-auto | gtags -f -

from your home directory or wherever you keep source code. If you are
making large changes, it is useful to either set this up to run as a
cron job or to make a convenient alias so that refreshing is easy. Then
add the following to ``~/.emacs`` to enable gtags and specify key bindings.

.. code-block:: emacs

   (when (require 'gtags)
     (global-set-key (kbd "C-c f") 'gtags-find-file)
     (global-set-key (kbd "C-c .") 'gtags-find-tag)
     (global-set-key (kbd "C-c r") 'gtags-find-rtag)
     (global-set-key (kbd "C-c ,") 'gtags-pop-stack))
   (add-hook 'c-mode-common-hook
             '(lambda () (gtags-mode t))) ; Or add to existing hook

A more basic alternative to the GNU Global (``gtags``) approach that does not require adding packages is to use
the builtin ``etags`` feature.  First, run ``make alletags`` from the
PETSc home directory to generate the file ``$PETSC_DIR/TAGS``, and
then from within Emacs, run

.. code-block:: none

   M-x visit-tags-table

where ``M`` denotes the Emacs Meta key, and enter the name of the
``TAGS`` file. Then the command ``M-.`` will cause Emacs to find the
file and line number where a desired PETSc function is defined. Any
string in any of the PETSc files can be found with the command ``M-x tags-search``.
To find repeated occurrences, one can simply use ``M-,`` to find the next occurrence.

VS Code Users
~~~~~~~~~~~~~
`VS Code <https://code.visualstudio.com/>`_ (unlike :ref:`sec-visual-studio`, described below) is an open source editor with a rich extension ecosystem.
It has `excellent integration <https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd>`_ with clangd and will automatically pick up ``compile_commands.json`` as produced by a command such as ``bear make -B`` (see :ref:`sec-emacs`).
If you have no prior attachment to a specific code editor, we recommend trying VS Code.

Vi and Vim Users
~~~~~~~~~~~~~~~~
See the :ref:`sec-emacs` discussion above for configuration of clangd, which provides integrated development environment.

If users develop application codes using Vi or Vim the ``tags`` feature
can be used to search PETSc files quickly and efficiently. To use this
feature, one should first check if the file, ``$PETSC_DIR/CTAGS``
exists. If this file is not present, it should be generated by running
``make alletags`` from the PETSc home directory. Once the file
exists, from Vi/Vim the user should issue the command

.. code-block:: none

   :set tags=CTAGS

from the ``$PETSC_DIR`` directory and enter the name of the ``CTAGS``
file. Then the command “tag functionname” will cause Vi/Vim to find the
file and line number where a desired PETSc function is defined. See, `online tutorials <http://www.yolinux.com/TUTORIALS/LinuxTutorialAdvanced_vi.html>`_
for additional Vi/Vim options that allow searches, etc. It is also
possible to use GNU Global with Vim; see the description for Emacs
above.

Eclipse Users
~~~~~~~~~~~~~

If you are interested in developing code that uses PETSc from Eclipse or
developing PETSc in Eclipse and have knowledge of how to do indexing and
build libraries in Eclipse, please contact us at
`petsc-dev@mcs.anl.gov <mailto:petsc-dev@mcs.anl.gov>`_.

One way to index and build PETSc in Eclipse is as follows.

#. Open
   “File\ :math:`\rightarrow`\ Import\ :math:`\rightarrow`\ Git\ :math:`\rightarrow`\ Projects
   from Git”. In the next two panels, you can either add your existing
   local repository or download PETSc from Bitbucket by providing the
   URL. Most Eclipse distributions come with Git support. If not,
   install the EGit plugin. When importing the project, select the
   wizard “Import as general project”.

#. Right-click on the project (or the “File” menu on top) and select
   “New :math:`\rightarrow` Convert to a C/C++ Project (Adds C/C++
   Nature)”. In the setting window, choose “C Project” and specify the
   project type as “Shared Library”.

#. Right-click on the C project and open the “Properties” panel. Under
   “C/C++ Build :math:`\rightarrow` Builder Settings”, set the Build
   directory to ``$PETSC_DIR`` and make sure “Generate Makefiles
   automatically” is unselected. Under the section “C/C++
   General\ :math:`\rightarrow`\ Paths and Symbols”, add the PETSc paths
   to “Includes”.

 .. code-block:: none

        $PETSC_DIR/include
        $PETSC_DIR/$PETSC_ARCH/include

   Under the section “C/C++ General\ :math:`\rightarrow`\ index”, choose
   “Use active build configuration”.

#. Configure PETSc normally outside Eclipse to generate a makefile and
   then build the project in Eclipse. The source code will be parsed by
   Eclipse.

If you launch Eclipse from the Dock on Mac OS X, ``.bashrc`` will not be
loaded (a known OS X behavior, for security reasons). This will be a
problem if you set the environment variables ``$PETSC_DIR`` and
``$PETSC_ARCH`` in ``.bashrc``. A solution which involves replacing the
executable can be found at
```/questions/829749/launch-mac-eclipse-with-environment-variables-set`` </questions/829749/launch-mac-eclipse-with-environment-variables-set>`__.
Alternatively, you can add ``$PETSC_DIR`` and ``$PETSC_ARCH`` manually
under “Properties :math:`\rightarrow` C/C++ Build :math:`\rightarrow`
Environment”.

To allow an Eclipse code to compile with the PETSc include files and
link with the PETSc libraries, a PETSc user has suggested the following.

#. Right-click on your C project and select “Properties
   :math:`\rightarrow` C/C++ Build :math:`\rightarrow` Settings”

#. A new window on the righthand side appears with various settings
   options. Select “Includes” and add the required PETSc paths,

.. code-block:: none

      $PETSC_DIR/include
      $PETSC_DIR/$PETSC_ARCH/include

#. Select “Libraries” under the header Linker and set the library search
   path:

.. code-block:: none

      $PETSC_DIR/$PETSC_ARCH/lib

   and the libraries, for example

.. code-block:: none

      m, petsc, stdc++, mpichxx, mpich, lapack, blas, gfortran, dl, rt,gcc_s, pthread, X11

Another PETSc user has provided the following steps to build an Eclipse
index for PETSc that can be used with their own code, without compiling
PETSc source into their project.

#. In the user project source directory, create a symlink to the PETSC
   ``src/`` directory.

#. Refresh the project explorer in Eclipse, so the new symlink is
   followed.

#. Right-click on the project in the project explorer, and choose “Index
   :math:`\rightarrow` Rebuild”. The index should now be built.

#. Right-click on the PETSc symlink in the project explorer, and choose
   “Exclude from build...” to make sure Eclipse does not try to compile
   PETSc with the project.

For further examples of using Eclipse with a PETSc-based application,
see the documentation for LaMEM [#lamem]_.

Qt Creator Users
~~~~~~~~~~~~~~~~

This information was provided by Mohammad Mirzadeh. The Qt Creator IDE
is part of the Qt SDK, developed for cross-platform GUI programming
using C++. It is available under GPL v3, LGPL v2 and a commercial
license and may be obtained, either as part of the Qt SDK or as
stand-alone software. It supports
automatic makefile generation using cross-platform ``qmake`` and
``cmake`` build systems as well as allowing one to import projects based
on existing, possibly hand-written, makefiles. Qt Creator has a visual
debugger using GDB and LLDB (on Linux and OS X) or Microsoft’s CDB (on
Windows) as backends. It also has an interface to Valgrind’s “memcheck”
and “callgrind” tools to detect memory leaks and profile code. It has
built-in support for a variety of version control systems including git,
mercurial, and subversion. Finally, Qt Creator comes fully equipped with
auto-completion, function look-up, and code refactoring tools. This
enables one to easily browse source files, find relevant functions, and
refactor them across an entire project.

Creating a Project
^^^^^^^^^^^^^^^^^^

When using Qt Creator with ``qmake``, one needs a ``.pro`` file. This
configuration file tells Qt Creator about all build/compile options and
locations of source files. One may start with a blank ``.pro`` file and
fill in configuration options as needed. For example:

.. code-block:: none

   # The name of the application executable
   TARGET = ex1

   # There are two ways to add PETSc functionality
   # 1-Manual: Set all include path and libs required by PETSc
   PETSC_INCLUDE = path/to/petsc_includes # e.g. obtained via running `make getincludedirs'
   PETSC_LIBS = path/to/petsc_libs # e.g. obtained via running `make getlinklibs'

   INCLUDEPATH += $$PETSC_INCLUDES
   LIBS += $$PETSC_LIBS

   # 2-Automatic: Use the PKGCONFIG funtionality
   # NOTE: PETSc.pc must be in the pkgconfig path. You might need to adjust PKG_CONFIG_PATH
   CONFIG += link_pkgconfig
   PKGCONFIG += PETSc

   # Set appropriate compiler and its flags
   QMAKE_CC = path/to/mpicc
   QMAKE_CXX = path/to/mpicxx # if this is a cpp project
   QMAKE_LINK = path/to/mpicxx # if this is a cpp project

   QMAKE_CFLAGS   += -O3 # add extra flags here
   QMAKE_CXXFLAGS += -O3
   QMAKE_LFLAGS   += -O3

   # Add all files that must be compiled
   SOURCES += ex1.c source1.c source2.cpp

   HEADERS += source1.h source2.h

   # OTHER_FILES are ignored during compilation but will be shown in file panel in Qt Creator
   OTHER_FILES += \
       path/to/resource_file \
       path/to/another_file

In this example, keywords include:

-  ``TARGET``: The name of the application executable.

-  ``INCLUDEPATH``: Used at compile time to point to required include
   files. Essentially, it is used as an ``-I \$\$INCLUDEPATH`` flag for
   the compiler. This should include all application-specific header
   files and those related to PETSc (which may be found via
   ``make getincludedirs``).

-  ``LIBS``: Defines all required external libraries to link with the
   application. To get PETSc’s linking libraries, use
   ``make getlinklibs``.

-  ``CONFIG``: Configuration options to be used by ``qmake``. Here, the
   option ``link_pkgconfig`` instructs ``qmake`` to internally use
   ``pkgconfig`` to resolve ``INCLUDEPATH`` and ``LIBS`` variables.

-  ``PKGCONFIG``: Name of the configuration file (the ``.pc`` file –
   here ``PETSc.pc``) to be passed to ``pkgconfig``. Note that for this
   functionality to work, ``PETSc.pc`` must be in path which might
   require adjusting the ``PKG_CONFIG_PATH`` enviroment variable. For
   more information see
   `the Qt Creator documentation <https://doc.qt.io/qtcreator/creator-build-settings.html>`__.

-  ``QMAKE_CC`` and ``QMAKE_CXX``: Define which C/C++ compilers use.

-  ``QMAKE_LINK``: Defines the proper linker to be used. Relevant if
   compiling C++ projects.

-  ``QMAKE_CFLAGS``, ``QMAKE_CXXFLAGS`` and ``QMAKE_LFLAGS``: Set the
   corresponding compile and linking flags.

-  ``SOURCES``: Source files to be compiled.

-  ``HEADERS``: Header files required by the application.

-  ``OTHER_FILES``: Other files to include (source, header, or any other
   extension). Note that none of the source files placed here are
   compiled.

More options can be included in a ``.pro`` file; see
https://doc.qt.io/qt-5/qmake-project-files.html. Once the ``.pro`` file
is generated, the user can simply open it via Qt Creator. Upon opening,
one has the option to create two different build options, debug and
release, and switch between the two. For more information on using the
Qt Creator interface and other more advanced aspects of the IDE, refer
to https://www.qt.io/qt-features-libraries-apis-tools-and-ide/

.. _sec-visual-studio:

Visual Studio Users
~~~~~~~~~~~~~~~~~~~

To use PETSc from MS Visual Studio, one would have to compile a PETSc
example with its corresponding makefile and then transcribe all compiler
and linker options used in this build into a Visual Studio project file,
in the appropriate format in Visual Studio project settings.

XCode Users (The Apple GUI Development System)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mac OS X
^^^^^^^^

Follow the instructions in ``$PETSC_DIR/systems/Apple/OSX/bin/makeall``
to build the PETSc framework and documentation suitable for use in
XCode.

You can then use the PETSc framework in
``$PETSC_DIR/arch-osx/PETSc.framework`` in the usual manner for Apple
frameworks. See the examples in
``$PETSC_DIR/systems/Apple/OSX/examples``. When working in XCode, things
like function name completion should work for all PETSc functions as
well as MPI functions. You must also link against the Apple
``Accelerate.framework``.

iPhone/iPad iOS
^^^^^^^^^^^^^^^

Follow the instructions in
``$PETSC_DIR/systems/Apple/iOS/bin/iosbuilder.py`` to build the PETSc
library for use on the iPhone/iPad.

You can then use the PETSc static library in
``$PETSC_DIR/arch-osx/libPETSc.a`` in the usual manner for Apple
libraries inside your iOS XCode projects; see the examples in
``$PETSC_DIR/systems/Apple/iOS/examples``. You must also link against
the Apple ``Accelerate.framework``.

.. rubric:: Footnotes

.. [#saws]
   `Saws wiki on Bitbucket <https://bitbucket.org/saws/saws/wiki/Home>`__

.. [#cxx_note]
   Note that this option is not required to use PETSc with C++

.. [#lamem]
   ``doc/`` at https://bitbucket.org/bkaus/lamem

.. [#json]
   JSON is a subset of YAML
