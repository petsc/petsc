Basic Object Design and Implementation
======================================

PETSc is designed by using strong data encapsulation. Hence, any
collection of data (for instance, a sparse matrix) is stored in a way
that is completely private from the application code. The application
code can manipulate the data only through a well-defined interface,
since it does *not* "know" how the data is stored internally.

Introduction
------------

PETSc is designed around several classes including ``Vec`` (vectors) and
``Mat`` (matrices, both dense and sparse). Each class is implemented by
using a C ``struct`` that contains the data and function pointers for
operations on the data (much like virtual functions in C++ classes).
Each class consists of three parts:

A (small) common part shared by all PETSc classes (for example, both
``KSP`` and ``PC`` have this same header).

Another common part shared by all PETSc implementations of the class
(for example, both ``KSPGMRES`` and ``KSPCG`` have this common
subheader).

A private part used by only one particular implementation written in
PETSc.

For example, all matrix (``Mat``) classes share a function table of
operations that may be performed on the matrix; all PETSc matrix
implementations share some additional data fields, including matrix
parallel layout, while a particular matrix implementation in PETSc (say
compressed sparse row) has its own data fields for storing the actual
matrix values and sparsity pattern. This will be explained in more
detail in the following sections. New class implementations *must* use
the PETSc common part.

We will use ``<class>_<implementation>`` to denote the actual source code
and data structures used for a particular implementation of an object
that has the ``<class>`` interface.

Organization of the Source Code
-------------------------------

Each class has the following organization.

Its own, application-public, include file ``include/petsc<class>.h``.

Its own directory, ``src/<class>`` or ``src/<package>/<class>``.

A data structure defined in the file
``include/petsc/private/<class>impl.h``. This data structure is shared
by all the different PETSc implementations of the class. For example,
for matrices it is shared by dense, sparse, parallel, and sequential
formats.

An abstract interface that defines the application-callable functions
for the class. These are defined in the directory
``src/<class>/interface``. This is how polymorphism is supported with
code that implements the abstract interface to the operations on the
object. Essentially, these routines do some error checking of arguments
and logging of profiling information and then call the function
appropriate for the particular implementation of the object. The name of
the abstract function is ``<class>Operation``, for instance,
``MatMult()`` or ``PCCreate(``), while the name of a particular
implementation is ``<class>Operation_<implementation>``, for instance,
``MatMult_SeqAIJ()`` or ``PCCreate_ILU()``. These naming conventions are
used to simplify code maintenance (also see Section [sec:stylenames]).

One or more actual implementations of the class (for example, sparse
uniprocessor and parallel matrices implemented with the AIJ storage
format). These are each in a subdirectory of ``src/<class>/impls``.
Except in rare circumstances, data structures defined here should not be
referenced from outside this directory.

Each type of object (for instance, a vector) is defined in its own
public include file, by ``typedef _p_<class>* <class>``; (for example,
``typedef _p_Vec* Vec;``). This organization allows the compiler to
perform type checking on all subroutine calls while at the same time
completely removing the details of the implementation of ``_p_<class>``
from the application code. This capability is extremely important
because it allows the library internals to be changed without altering
or recompiling the application code.

Common Object Header
--------------------

All PETSc/PETSc objects have the following common header structures
defined in
`include/petsc/private/petscimpl.h <../../include/petsc/private/petscimpl.h.html>`__:

.. code-block::

    typedef struct {
      PetscErrorCode (*getcomm)(PetscObject,MPI_Comm*);
      PetscErrorCode (*view)(PetscObject,Viewer);
      PetscErrorCode (*destroy)(PetscObject);
      PetscErrorCode (*query)(PetscObject,const char*,PetscObject*);
      PetscErrorCode (*compose)(PetscObject,const char*,PetscObject);
      PetscErrorCode (*composefunction)(PetscObject,const char*,void(*)(void));
      PetscErrorCode (*queryfunction)(PetscObject,const char*,void (**)(void));
    } PetscOps;

.. code-block::

    struct _p_<class> {
      PetscClassId     classid;
      PetscOps         *bops;
      <class>Ops       *ops;
      MPI_Comm         comm;
      PetscLogDouble   flops,time,mem;
      int              id;
      int              refct;
      int              tag;
      DLList           qlist;
      OList            olist;
      char             *type_name;
      PetscObject      parent;
      char             *name;
      char             *prefix;
      void             *cpp;
      void             **fortran_func_pointers;
      ..........
      CLASS-SPECIFIC DATASTRUCTURES
    };

Here ``<class>ops`` is a function table (like the ``PetscOps`` above)
that contains the function pointers for the operations specific to that
class. For example, the PETSc vector class object operations in
`include/petsc/private/vecimpl.h <../../include/petsc/private/vecimpl.h.html>`__
include the following.

.. code-block::

    typedef struct _VecOps* VecOps;
    struct _VecOps {
      PetscErrorCode (*duplicate)(Vec,Vec*); /* get single vector */
      PetscErrorCode (*duplicatevecs)(Vec,PetscInt,Vec**); /* get array of vectors */
      PetscErrorCode (*destroyvecs)(PetscInt,Vec[]); /* free array of vectors */
      PetscErrorCode (*dot)(Vec,Vec,PetscScalar*); /* z = x^H * y */
      PetscErrorCode (*mdot)(Vec,PetscInt,const Vec[],PetscScalar*); /* z[j] = x dot y[j] */
      PetscErrorCode (*norm)(Vec,NormType,PetscReal*); /* z = sqrt(x^H * x) */
      PetscErrorCode (*tdot)(Vec,Vec,PetscScalar*); /* x'*y */
      PetscErrorCode (*mtdot)(Vec,PetscInt,const Vec[],PetscScalar*);/* z[j] = x dot y[j] */
      PetscErrorCode (*scale)(Vec,PetscScalar);  /* x = alpha * x   */
      PetscErrorCode (*copy)(Vec,Vec); /* y = x */
      PetscErrorCode (*set)(Vec,PetscScalar); /* y = alpha  */
      PetscErrorCode (*swap)(Vec,Vec); /* exchange x and y */
      PetscErrorCode (*axpy)(Vec,PetscScalar,Vec); /* y = y + alpha * x */
      PetscErrorCode (*axpby)(Vec,PetscScalar,PetscScalar,Vec); /* y = alpha * x + beta * y*/
      PetscErrorCode (*maxpy)(Vec,PetscInt,const PetscScalar*,Vec*); /* y = y + alpha[j] x[j] */
      ... (AND SO ON) ...
    };

.. code-block::

    struct _p_Vec {
      PetscClassId           classid;
      PetscOps               *bops;
      VecOps                 *ops;
      MPI_Comm               comm;
      PetscLogDouble         flops,time,mem;
      int                    id;
      int                    refct;
      int                    tag;
      DLList                 qlist;
      OList                  olist;
      char                   *type_name;
      PetscObject            parent;
      char                   *name;
      char                   *prefix;
      void                   **fortran_func_pointers;
      void                   *data;     /* implementation-specific data */
      PetscLayout            map;
      ISLocalToGlobalMapping mapping;   /* mapping used in VecSetValuesLocal() */
    };

Each PETSc object begins with a ``PetscClassId``, which is used for
error checking. Each different class of objects has its value for
``classid``; these are used to distinguish between classes. When a new
class is created you need to call

.. code-block::

    PetscClassIdRegister(const char *classname,PetscClassId *classid);

For example,

.. code-block::

    PetscClassIdRegister("index set",&IS_CLASSID);

you can verify that an object is valid of a particular class with
``PetscValidHeaderSpecific``, for example,

.. code-block::

    PetscValidHeaderSpecific(x,VEC_CLASSID,1);

The third argument to this macro indicates the position in the calling
sequence of the function the object was passed in. This is to generate
more complete error messages.

To check for an object of any type, use

.. code-block::

    PetscValidHeader(x,1);

Common Object Functions
-----------------------

Several routines are provided for manipulating data within the header.
These include the specific functions in the PETSc common function table.
The function pointers are not called directly; rather you should call
``PetscObjectFunctionName()``, where ``FunctionName`` is one of the
functions listed below with the first letter of each word capitalized.

``getcomm(PetscObject,MPI_Comm*)`` obtains the MPI communicator
associated with this object.

``view(PetscObject,PetscViewer)`` allows you to store or visualize the
data inside an object. If the Viewer is NULL, then it should cause the
object to print information on the object to textttstdout.

``destroy(PetscObject)`` causes the reference count of the object to be
decreased by one or the object to be destroyed and all memory used by
the object to be freed when the reference count drops to zero. If the
object has any other objects composed with it, they are each sent a
``destroy()``; that is, the ``destroy()`` function is called on them
also.

``compose(PetscObject,const char *name,PetscObject)`` associates the
second object with the first object and increases the reference count of
the second object. If an object with the same name was previously
composed, that object is dereferenced and replaced with the new object.
If the second object is NULL and an object with the same name has
already been composed, that object is dereferenced (the ``destroy()``
function is called on it, and that object is removed from the first
object). This is a way to remove, by name, an object that was previously
composed.

``query(PetscObject,const char *name,PetscObject*)`` retrieves an object
that was previously composed with the first object via
``PetscObjectCompose()``. It retrieves a NULL if no object with that
name was previously composed.

``composefunction(PetscObject,const char *name,void *func)`` associates
a function pointer with an object. If the object already had a composed
function with the same name, the old one is replaced. If ``func`` is
``NULL``, the existing function is removed from the object. The string
``name`` is the character string name of the function.

For example, ``fname`` may be ``PCCreate_LU``.

``queryfunction(PetscObject,const char *name,void **func)`` retrieves a
function pointer that was associated with the object via
``PetscObjectComposeFunction()``. If dynamic libraries are used, the
function is loaded into memory at this time (if it has not been
previously loaded), not when the ``composefunction()`` routine was
called.

Since the object composition allows one to compose PETSc objects *only*
with PETSc objects rather than any arbitrary pointer, PETSc provides the
convenience object ``PetscContainer``, created with the routine
``PetscContainerCreate(MPI_Comm,PetscContainer*)``, to allow wrapping any
kind of data into a PETSc object that can then be composed with a PETSc
object.

Object Function Implementation
------------------------------

This section discusses how PETSc implements the ``compose()``,
``query()``, ``composefunction()``, and ``queryfunction()`` functions
for its object implementations. Other PETSc-compatible class
implementations are free to manage these functions in any manner; but
unless there is a specific reason, they should use the PETSc defaults so
that the library writer does not have to “reinvent the wheel.”

Compose and Query Objects
~~~~~~~~~~~~~~~~~~~~~~~~~

In
`src/sys/objects/olist.c <../../src/sys/objects/olist.c.html>`__,
PETSc defines a C ``struct``

.. code-block::

      typedef struct _PetscObjectList* PetscObjectList;
      struct _PetscObjectList {
          char             name[128];
          PetscObject      obj;
          PetscObjectList  next;
      };

from which linked lists of composed objects may be constructed. The
routines to manipulate these elementary objects are

.. code-block::

    int PetscObjectListAdd(PetscObjectList *fl,const char *name,PetscObject obj);
    int PetscObjectListDestroy(PetscObjectList *fl);
    int PetscObjectListFind(PetscObjectList fl,const char *name,PetscObject *obj)
    int PetscObjectListDuplicate(PetscObjectList fl,PetscObjectList *nl);

The function ``PetscObjectListAdd()`` will create the initial
PetscObjectList if the argument ``fl`` points to a NULL.

The PETSc object ``compose()`` and ``query()`` functions are as follows
(defined in
`src/sys/objects/inherit.c <../../src/sys/objects/inherit.c.html>`__).

.. code-block::

    PetscErrorCode PetscObjectCompose_Petsc(PetscObject obj,const char *name,PetscObject ptr)
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      PetscObjectListAdd(&obj->olist,name,ptr);
      PetscFunctionReturn(0);
    }

    PetscErrorCode PetscObjectQuery_Petsc(PetscObject obj,const char *name,PetscObject *ptr)
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      PetscObjectListFind(obj->olist,name,ptr);
      PetscFunctionReturn(0);
    }

Compose and Query Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSc allows you to compose functions by specifying a name and function
pointer. In
`src/sys/dll/reg.c <../../src/sys/dll/reg.c.html>`__,
PETSc defines the following linked list structure.

.. code-block::

    struct _n_PetscFunctionList {
      void              (*routine)(void);    /* the routine */
      char              *name;               /* string to identify routine */
      PetscFunctionList next;                /* next pointer */
      PetscFunctionList next_list;           /* used to maintain list of all lists for freeing */
    };

Each PETSc object contains a ``PetscFunctionList`` object. The
``composefunction()`` and ``queryfunction()`` are given by the
following.

.. code-block::

    PetscErrorCode PetscObjectComposeFunction_Petsc(PetscObject obj,const char *name,void *ptr)
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      PetscFunctionListAdd(&obj->qlist,name,fname,ptr);
      PetscFunctionReturn(0);
    }

    PetscErrorCode PetscObjectQueryFunction_Petsc(PetscObject obj,const char *name,void (**ptr)(void))
    {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      PetscFunctionListFind(obj->qlist,name,ptr);
      PetscFunctionReturn(0);
    }

In addition to using the ``PetscFunctionList`` mechanism to compose
functions into PETSc objects, it is also used to allow registration of
new class implementations; for example, new preconditioners.

Simple PETSc Objects
~~~~~~~~~~~~~~~~~~~~

Some simple PETSc objects do not need ``PETSCHEADER`` and the associated
functionality. These objects are internally named as ``_n_<class>`` as
opposed to ``_p_<class>``, for example, ``_n_PetscTable`` vs ``_p_Vec``.

PETSc Packages
--------------

The PETSc source code is divided into the following library-level
packages: ``sys``, ``Vec``, ``Mat``, ``DM``, ``KSP``, ``SNES``, ``TS``,
``TAO``. Each of these has a directory under the ``src`` directory in
the PETSc tree and, optionally, can be compiled into separate libraries.
Each package defines one or more classes; for example, the ``KSP``
package defines the ``KSP`` and ``PC`` classes, as well as several
utility classes. In addition, each library-level package may contain
several class-level packages associated with individual classes in the
library-level package. In general, most “important” classes in PETSc
have their own class level package. Each package provides a registration
function ``XXXInitializePackage()``, for example
``KSPInitializePackage()``, which registers all the classes and events
for that package. Each package also registers a finalization routine,
``XXXFinalizePackage()``, that releases all the resources used in
registering the package, using ``PetscRegisterFinalize()``. The
registration for each package is performed “on demand” the first time a
class in the package is utilized. This is handled, for example, with
code such as

.. code-block::

    PetscErrorCode  VecCreate(MPI_Comm comm, Vec *vec)
    {
      Vec            v;

      PetscFunctionBegin;
      PetscValidPointer(vec,2);
      *vec = NULL;
      VecInitializePackage();
      ...
