# Basic Object Design and Implementation

PETSc is designed by using strong data encapsulation. Hence, any
collection of data (for instance, a sparse matrix) is stored in a way
that is completely private from the application code. The application
code can manipulate the data only through a well-defined interface,
since it does *not* "know" how the data is stored internally.

## Introduction

PETSc is designed around several classes including `Vec` (vectors) and
`Mat` (matrices, both dense and sparse). Each class is implemented by
using a C `struct` that contains the data and function pointers for
operations on the data (much like virtual functions in C++ classes).
Each class consists of three parts:

A (small) common part shared by all PETSc classes (for example, both
`KSP` and `PC` have this same header).

Another common part shared by all PETSc implementations of the class
(for example, both `KSPGMRES` and `KSPCG` have this common
subheader).

A private part used by only one particular implementation written in
PETSc.

For example, all matrix (`Mat`) classes share a function table of
operations that may be performed on the matrix; all PETSc matrix
implementations share some additional data fields, including matrix
parallel layout, while a particular matrix implementation in PETSc (say
compressed sparse row) has its own data fields for storing the actual
matrix values and sparsity pattern. This will be explained in more
detail in the following sections. New class implementations *must* use
the PETSc common part.

We will use `<class>_<implementation>` to denote the actual source code
and data structures used for a particular implementation of an object
that has the `<class>` interface.

## Organization of the Source Code

Each class has the following organization.

Its own, application-public, include file `include/petsc<class>.h`.

Its own directory, `src/<class>` or `src/<package>/<class>`.

A data structure defined in the file
`include/petsc/private/<class>impl.h`. This data structure is shared
by all the different PETSc implementations of the class. For example,
for matrices it is shared by dense, sparse, parallel, and sequential
formats.

An abstract interface that defines the application-callable functions
for the class. These are defined in the directory
`src/<class>/interface`. This is how polymorphism is supported with
code that implements the abstract interface to the operations on the
object. Essentially, these routines do some error checking of arguments
and logging of profiling information and then call the function
appropriate for the particular implementation of the object. The name of
the abstract function is `<class>Operation`, for instance,
`MatMult()` or `PCCreate()`, while the name of a particular
implementation is `<class>Operation_<implementation>`, for instance,
`MatMult_SeqAIJ()` or `PCCreate_ILU()`. These naming conventions are
used to simplify code maintenance (also see {ref}`style`).

One or more actual implementations of the class (for example, sparse
uniprocessor and parallel matrices implemented with the AIJ storage
format). These are each in a subdirectory of `src/<class>/impls`.
Except in rare circumstances, data structures defined here should not be
referenced from outside this directory.

Each type of object (for instance, a vector) is defined in its own
public include file, by `typedef _p_<class>* <class>`; (for example,
`typedef _p_Vec* Vec;`). This organization allows the compiler to
perform type checking on all subroutine calls while at the same time
completely removing the details of the implementation of `_p_<class>`
from the application code. This capability is extremely important
because it allows the library internals to be changed without altering
or recompiling the application code.

## Common Object Header

All PETSc objects (derived from the base class `PetscObject`) have the following common header structures
defined in
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petsc/private/petscimpl.h.html">include/petsc/private/petscimpl.h</a>

```{literalinclude} /../include/petsc/private/petscimpl.h
:end-at: } PetscOps;
:language: c
:start-at: typedef struct {
```

```{literalinclude} /../include/petsc/private/petscimpl.h
:end-at: ObjectOps      ops[1]
:language: c
:start-at: '#define PETSCHEADER(ObjectOps)'
```

Here `ObjectOps` is a function table (like the `PetscOps` above)
that contains the function pointers for the operations specific to that
class. For example, the PETSc vector class object operations in
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petsc/private/vecimpl.h.html">include/petsc/private/vecimpl.h</a>
include the following.

```{literalinclude} /../include/petsc/private/vecimpl.h
:end-at: PetscErrorCode (*maxpy)
:language: c
:start-at: typedef struct _VecOps *VecOps;
```

```{literalinclude} /../include/petsc/private/vecimpl.h
:end-at: };
:language: c
:start-at: struct _p_Vec {
```

Each PETSc object contains a `PetscClassId`, which is used for
error checking. Each class has a unique `classid`; these values distinguish
between classes. When a new
class is created you need to call

```
PetscErrorCode PetscClassIdRegister(const char[], PetscClassId *);
```

For example,

```
PetscClassIdRegister("index set",&IS_CLASSID);
```

you can verify that an object is valid of a particular class with
`PetscValidHeaderSpecific`, for example,

```
PetscValidHeaderSpecific(x,VEC_CLASSID,1);
```

The third argument to this macro indicates the position in the calling
sequence of the function the object was passed in. This is to generate
more complete error messages.

To check for an object of any type, use

```
PetscValidHeader(x,1);
```

The `obj->ops` functions provide implementations of the standard methods of the object class. Each type
of the class may have different function pointers in the array. Subtypes sometimes replace some of the
function pointers of the parent, so they play the role of virtual methods in C++.

PETSc code that calls these function pointers should be done via

```
PetscUseTypeMethod(obj,method,other arguments);
PetscTryTypeMethod(obj,method,other arguments);
```

For example,

```
PetscErrorCode XXXOp(XXX x,YYY y)
{
  PetscFunctionBegin;
  PetscUseTypeMethod(x,op,y);
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

The `Try` variant skips the function call if the method has not been set while the `Use` version generates an error in that case.

See also, `PetscUseMethod()`, and `PetscTryMethod()`.

## Common Object Functions

Several routines manipulate data stored in the common object header.
Application code calls these routines instead of accessing the header
directly.

`PetscObjectGetComm()` returns the communicator stored in the object.

`PetscObjectView()` dispatches through the common `view` operation to
display or store information about the object. If the `PetscViewer` is
`NULL`, PETSc uses an ASCII viewer for `stdout`.

`PetscObjectDestroy()` dispatches through the common `destroy` operation.
The class-specific implementation manages reference counting and releases
the object when its reference count reaches zero.

`PetscObjectCompose()` associates another PETSc object with a name in the
object's composed-object list. It replaces an existing association with the
same name and removes the association when the supplied object is `NULL`.
`PetscObjectQuery()` retrieves an object from this list without increasing
its reference count and returns `NULL` when the name is not present.

`PetscObjectComposeFunction()` associates a function pointer with a name in
the object's composed-function list. It replaces an existing association
and removes the association when the function pointer is `NULL`.
`PetscObjectQueryFunction()` retrieves a function pointer from this list.

Since the object composition allows one to compose PETSc objects
with PETSc objects, PETSc provides the
convenience object `PetscContainer`, created with the routine
`PetscContainerCreate(MPI_Comm,PetscContainer*)`, to allow wrapping any
kind of data into a PETSc object that can then be composed with a PETSc
object. One can also use `PetscObjectContainerCompose()` and `PetscObjectContainerQuery()` to compose
arbitrary pointers with a PETSc object.

## Object Function Implementation

This section discusses how PETSc implements the `compose()`,
`query()`, `composefunction()`, and `queryfunction()` functions
for its object implementations. Other PETSc-compatible class
implementations are free to manage these functions in any manner; but
unless there is a specific reason, they should use the PETSc defaults so
that the library writer does not have to “reinvent the wheel.”

### Compose and Query Objects

PETSc defines the composed-object list in
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petsc/private/petscimpl.h.html">include/petsc/private/petscimpl.h</a>

```{literalinclude} /../include/petsc/private/petscimpl.h
:end-at: };
:language: c
:start-at: struct _n_PetscObjectList {
```

from which linked lists of composed objects may be constructed. The
routines to manipulate these elementary objects are

```
PetscErrorCode PetscObjectListAdd(PetscObjectList *fl, const char name[], PetscObject obj);
PetscErrorCode PetscObjectListDestroy(PetscObjectList *ifl);
PetscErrorCode PetscObjectListFind(PetscObjectList fl, const char name[], PetscObject *obj);
PetscErrorCode PetscObjectListDuplicate(PetscObjectList fl, PetscObjectList *nl);
```

The function `PetscObjectListAdd()` will create the initial
PetscObjectList if the argument `fl` points to a NULL.

The `PetscObjectCompose()` and `PetscObjectQuery()` functions are as follows
(defined in
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/sys/objects/inherit.c.html">src/sys/objects/inherit.c</a>

```{literalinclude} /../src/sys/objects/inherit.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode PetscObjectCompose(
```

```{literalinclude} /../src/sys/objects/inherit.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode PetscObjectQuery(
```

### Compose and Query Functions

PETSc allows you to compose functions by specifying a name and function
pointer. Each PETSc object contains a `PetscFunctionList` object. The
`PetscObjectComposeFunction()` and `PetscObjectQueryFunction()` are given by the
following.

```{literalinclude} /../src/sys/objects/inherit.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode PetscObjectComposeFunction_Private(
```

```{literalinclude} /../src/sys/objects/inherit.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PETSC_EXTERN PetscErrorCode PetscObjectQueryFunction_Private(
```

In addition to using the `PetscFunctionList` mechanism to compose
functions into PETSc objects, it is also used to allow registration of
new class implementations; for example, new preconditioners.

PETSc code that calls composed functions should be done via

```
PetscUseMethod(obj,"method",(Argument types),(argument variables));
PetscTryMethod(obj,"method",(Argument types),(argument variables));
```

For example,

```{literalinclude} /../src/ksp/ksp/impls/gmres/gmres.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode KSPGMRESSetRestart(
```

The `Try` variant skips the function call if the method has not been composed with
the object while the `Use` version generates an error in that case.
See also, `PetscUseTypeMethod()`, and `PetscTryTypeMethod()`.

### Simple PETSc Objects

Some simple PETSc objects do not need `PETSCHEADER` and the associated
functionality. These objects are internally named as `_n_<class>` as
opposed to `_p_<class>`, for example, `_n_PetscFunctionList` vs `_p_Vec`.

## PETSc Packages

The PETSc source code is divided into the following library-level
packages: `Sys`, `Vec`, `Mat`, `DM`, `KSP`, `SNES`, `TS`,
`Tao`. Each of these has a directory under the `src` directory in
the PETSc tree and, optionally, can be compiled into separate libraries.
Each package defines one or more classes; for example, the `KSP`
package defines the `KSP` and `PC` classes, as well as several
utility classes. In addition, each library-level package may contain
several class-level packages associated with individual classes in the
library-level package. In general, most “important” classes in PETSc
have their own class level package. Each package provides a registration
function `XXXInitializePackage()`, for example
`KSPInitializePackage()`, which registers all the classes and events
for that package. Each package also registers a finalization routine,
`XXXFinalizePackage()`, that releases all the resources used in
registering the package, using `PetscRegisterFinalize()`. The
registration for each package is performed “on demand” the first time a
class in the package is utilized. This is handled, for example, with
code such as

```{literalinclude} /../src/vec/vec/interface/veccreate.c
:append: '}'
:end-at: PetscFunctionReturn(PETSC_SUCCESS);
:language: c
:start-at: PetscErrorCode VecCreate(MPI_Comm comm, Vec *vec)
```
