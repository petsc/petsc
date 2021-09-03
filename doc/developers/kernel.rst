The PETSc Kernel
================

PETSc provides a variety of basic services for writing scalable,
component-based libraries; these are referred to as the PETSc kernel :cite:`bgms98`.
The source code for the kernel is in ``src/sys``. It contains systematic
support for

* managing PETSc types,
* error handling,
* memory management,
* profiling,
* object management,
* Fortran interfaces (see :cite:`BalayBrownKnepleyMcInnesSmith2015`)
* mechanism for generating appropriate citations for algorithms and software used in PETSc (see :cite:`knepley2013accurately`)
* file I/O,
* an options database, and
* objects and code for viewing, drawing, and displaying data and solver objects.

Each of these is discussed in a section below.

PETSc Types
-----------

For maximum flexibility, the basic data types ``int``, ``double``, and
so on are not used in PETSc source code. Rather, it has

* ``PetscScalar``,
* ``PetscInt``,
* ``PetscMPIInt``,
* ``PetscBLASInt``,
* ``PetscBool``, and
* ``PetscBT`` - bit storage of logical true and false.

``PetscInt`` can be set using ``configure`` to be either ``int`` (32
bit, the default) or ``long long`` (64 bit, with
``configure –with-64-bit-indices``) to allow indexing into very large
arrays. ``PetscMPIInt`` is used for integers passed to MPI as counts and
sizes. These are always ``int`` since that is what the MPI standard
uses. Similarly, ``PetscBLASInt`` is for counts, and so on passed to
BLAS and LAPACK routines. These are almost always ``int`` unless one is
using a special “64-bit integer” BLAS/LAPACK (this is available, for
example, with Intel’s MKL and OpenBLAS).

In addition, there are special types:

* ``PetscClassId``
* ``PetscErrorCode``
* ``PetscLogEvent``

These are currently always ``int``, but their use clarifies the code.

Implementation of Error Handling
--------------------------------

PETSc uses a “call error handler; then (depending on result) return
error code” model when problems are detected in the running code. The
public include file for error handling is
`include/petscerror.h <../../include/petscerror.h.html>`__,
and the source code for the PETSc error handling is in
``src/sys/error/``.

Simplified Interface
~~~~~~~~~~~~~~~~~~~~

The simplified macro-based interface consists of the following two
calls:

* ``SETERRQ(comm,error code,Error message);``
* ``CHKERRQ(ierr);``

The macro ``SETERRQ()`` is given by

.. literalinclude:: /../include/petscerror.h
   :language: c
   :start-at: #define SETERRQ
   :end-at: #define SETERRQ

It calls the error handler with the current function name and location:
line number, and file, plus an error code and an error message. Normally
``comm`` is ``PETSC_COMM_SELF``; it can be another communicator only if
one is absolutely sure the same error will be generated on all processes
in the communicator. This feature is to prevent the same error message
from being printed by many processes.

The macro ``CHKERRQ()`` is defined by

.. literalinclude:: /../include/petscerror.h
   :language: c
   :start-at: #define CHKERRQ
   :end-at: #define CHKERRQ

In addition to ``SETERRQ()``, the macros ``SETERRQ1()``, ``SETERRQ2()``,
``SETERRQ3()``, and ``SETERRQ4()`` allow one to provide additional
arguments to a formatted message string, for example,

::

    SETERRQ2(comm,PETSC_ERR,"Iteration overflow: its %D norm %g",its,(double)norm);

The reason for the numbered format is that C89 CPP macros cannot handle
a variable number of arguments.

Error Handlers
~~~~~~~~~~~~~~

The error-handling function ``PetscError()`` calls the “current” error
handler with the code

.. literalinclude:: /../src/sys/error/err.c
   :language: c
   :start-at: PetscErrorCode PetscError(
   :end-at: PetscFunctionReturn
   :append: }

You can set a new error handler with the command
``PetscPushErrorHandler()``, which maintains a linked list of error
handlers. The most recent error handler is removed via
``PetscPopErrorHandler()``.

PETSc provides several default error handlers:

* ``PetscTraceBackErrorHandler()``, the default;
* ``PetscAbortErrorHandler()``, called with ``-onerrorabort``, useful when running in the debugger;
* ``PetscReturnErrorHandler()``, which returns up the stack without printing error messages;
* ``PetscEmacsClientErrorHandler()``;
* ``PetscMPIAbortErrorHandler()``, which calls ``MPIAbort()`` after printing the error message; and
* ``PetscAttachDebuggerErrorHandler()``, called with ``-onerrorattachdebugger``.

Error Codes
~~~~~~~~~~~

The PETSc error handler takes an error code. The generic error codes are
defined in
`include/petscerror.h <../../include/petscerror.h.html>`__.
The same error code is used many times in the libraries. For example,
the error code ``PETSCERRMEM`` is used whenever a requested memory
allocation is not available.

Detailed Error Messages
~~~~~~~~~~~~~~~~~~~~~~~

In a modern parallel component-oriented application code, it does not
always make sense to simply print error messages to the terminal (and
more than likely there is no “terminal”, for example, with Microsoft
Windows or Apple iPad applications). PETSc provides the replaceable
function pointer

::

    (*PetscErrorPrintf)("Format",...);

which, by default, prints to standard out. Thus, error messages should
not be printed with ``printf()`` or ``fprintf()``. Rather, they should
be printed with ``(*PetscErrorPrintf)()``. You can direct all error
messages to ``stderr``, instead of the default ``stdout``, with the
command line option ``-erroroutputstderr``.

C++ Exceptions
~~~~~~~~~~~~~~

In PETSc code, when one calls C++ functions that do not return with an error code but might
instead throw C++ exceptions, one can use ``CHKERRCXX(func)``, which catches the exceptions
in *func* and then calls ``SETERRQ1()``.  The macro ``CHKERRCXX(func)`` is given by

.. literalinclude:: /../include/petscerror.h
   :language: c
   :start-at: #define CHKERRCXX
   :end-at: #define CHKERRCXX

Memory Management
-----------------

PETSc provides simple wrappers for the system ``malloc(), calloc()``,
and ``free()`` routines. The public interface for these is provided in
``petscsys.h``, while the implementation code is in ``src/sys/memory``.
The most basic interfaces are

.. literalinclude:: /../include/petscsys.h
   :language: c
   :start-at: #define PetscMalloc
   :end-at: #define PetscMalloc

.. literalinclude:: /../include/petscsys.h
   :language: c
   :start-at: #define PetscFree
   :end-at: #define PetscFree

.. literalinclude:: /../src/sys/memory/mal.c
   :language: c
   :start-at: PetscErrorCode PetscMallocA(
   :end-at: PetscFunctionReturn(0)
   :append: }

.. literalinclude:: /../src/sys/memory/mal.c
   :language: c
   :start-at: PetscErrorCode PetscFreeA(
   :end-at: PetscFunctionReturn(0)
   :append: }

which allow the use of any number of profiling and error-checking
wrappers for ``malloc(), calloc()``, and ``free()``. Both
``PetscMallocA()`` and ``PetscFreeA()`` call the function pointer values
``(*PetscTrMalloc)`` and ``(*PetscTrFree)``. ``PetscMallocSet()`` is
used to set these function pointers. The functions are guaranteed to
support requests for zero bytes of memory correctly. Freeing memory
locations also sets the pointer value to zero, preventing later code
from accidentally using memory that has been freed. All PETSc memory
allocation calls are memory aligned on at least double-precision
boundaries; the macro generated by configure ``PETSCMEMALIGN`` indicates
in bytes what alignment all allocations have. This can be controlled at
configure time with the option ``-with-memalign=<4,8,16,32,64>``.

``PetscMallocA()`` supports a request for up to 7 distinct memory
locations of possibly different types. This serves two purposes: it
reduces the number of system ``malloc()`` calls, thus potentially
increasing performance, and it clarifies in the code related memory
allocations that should be freed together.

The following macros are the preferred way to obtain and release memory
in the PETSc source code. They automatically manage calling
``PetscMallocA()`` and ``PetscFreeA()`` with the appropriate location
information.

.. literalinclude:: /../include/petscsys.h
   :language: c
   :start-at: #define PetscMalloc1
   :end-at: #define PetscMalloc1

.. literalinclude:: /../include/petscsys.h
   :language: c
   :start-at: #define PetscMalloc2
   :end-at: #define PetscMalloc2

...

.. literalinclude:: /../include/petscsys.h
   :language: c
   :start-at: #define PetscMalloc7
   :end-at: #define PetscMalloc7

.. literalinclude:: /../include/petscsys.h
   :language: c
   :start-at: #define PetscFree
   :end-at: #define PetscFree

.. literalinclude:: /../include/petscsys.h
   :language: c
   :start-at: #define PetscFree2
   :end-at: #define PetscFree2

...

.. literalinclude:: /../include/petscsys.h
   :language: c
   :start-at: #define PetscFree7
   :end-at: #define PetscFree7

Similar routines, ``PetscCalloc1()`` to ``PetscCalloc7()``, provide
memory initialized to zero. The size requests for these macros are in
number of data items requested, not in bytes. This decreases the number
of errors in the code since the compiler determines their sizes from the
object type instead of requiring the user to provide the correct value
with ``sizeof()``.

The routines ``PetscTrMallocDefault()`` and ``PetscTrFreeDefault()``,
which are set with the routine ``PetscSetUseTrMallocPrivate()`` (and are
used by default for the debug version of PETSc), provide simple logging
and error checking versions of memory allocation.

Implementation of Profiling
---------------------------

This section provides details about the implementation of event logging
and profiling within the PETSc kernel. The interface for profiling in
PETSc is contained in the file
`include/petsclog.h <../../include/petsclog.h.html>`__.
The source code for the profile logging is in ``src/sys/plog/``.

Profiling Object Creation and Destruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The creation of objects is profiled with the command
``PetscLogObjectCreate()``

::

    PetscLogObjectCreate(PetscObject h);

which logs the creation of any PETSc object. Just before an object is
destroyed, it should be logged with ``PetscLogObjectDestroy()``

::

    PetscLogObjectDestroy(PetscObject h);

These are called automatically by ``PetscHeaderCreate()`` and
``PetscHeaderDestroy()``, which are used in creating all objects
inherited from the basic object. Thus, these logging routines need never
be called directly.

If an object has a clearly defined parent object (for instance, when a
work vector is generated for use in a Krylov solver), this information
is logged with the command ``PetscLogObjectParent()``.

::

    PetscLogObjectParent(PetscObject parent,PetscObject child);

It is also useful to log information about the state of an object, as
can be done with the command

::

    PetscLogObjectState(PetscObject h,const char *format,...);

For example, for sparse matrices we usually log the matrix dimensions
and number of nonzeros.

Profiling Events
~~~~~~~~~~~~~~~~

Events are logged by using the pair ``PetscLogEventBegin()`` and ``PetscLogEventEnd()``.

This logging is usually done in the abstract interface file for the
operations, for example,
`src/mat/interface/matrix.c <../../src/mat/interface/matrix.c.html>`__.

Controlling Profiling
~~~~~~~~~~~~~~~~~~~~~

Routines that control the default profiling available in PETSc include
the following

* ``PetscLogDefaultBegin();``
* ``PetscLogAllBegin();``
* ``PetscLogDump(const char *filename);``
* ``PetscLogView(PetscViewer);``

These routines are normally called by the ``PetscInitialize()`` and
``PetscFinalize()`` routines when the option ``-logview`` is given.

References
----------

.. bibliography:: /../src/docs/tex/petsc.bib
   :filter: docname in docnames

.. bibliography:: /../src/docs/tex/petscapp.bib
   :filter: docname in docnames
