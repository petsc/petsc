/**
 \page dev-basic-object-design Basic Object Design


%PETSc is designed using strong data encapsulation.  Hence,
any collection of data (for instance, a sparse matrix) is stored in
a way that is completely private from the application code. The application
code can manipulate the data only through a well-defined interface, as it
does **not** ``know'' how the data is stored internally.

\section dev-basic-object-design-introduction Introduction

%PETSc is designed around several classes (e.g. Vec (vectors),
Mat (matrices, both dense and sparse)). These classs are each
implemented using C `struct`s, that contain the data and function pointers
for operations on the data (much like virtual functions in classes in C++).
Each class consists of three parts:
  - a (small) common part shared by all %PETSc compatible libraries.
  - another common part shared by all %PETSc implementations of the class and
  - a private part used by only one particular implementation written in %PETSc.

For example, all matrix (Mat) classs share a function table of operations that
may be performed on the matrix; all %PETSc matrix implementations share some additional
data fields,including matrix size; while a particular matrix implementation in %PETSc
(say compressed sparse row) has its own data fields for storing the actual
matrix values and sparsity pattern. This will be explained in more detail
in the following sections. People providing new class implementations {\bf must}
use the %PETSc common part.


We will use `<class>_<implementation>` to denote the actual source code and
data structures used for a particular implementation of an object that has the
`<class>` interface.

\section dev-basic-object-design-organization Organization of the Source Code

Each class has
  - Its own, application public, include file `include/petsc<class>.h`
  - Its own directory, `src/<class>`
  - A data structure defined in the file `include/petsc-private/<class>impl.h`.
      This data structure is shared by all the different %PETSc implementations of the
      class. For example, for matrices it is shared by dense,
      sparse, parallel, and sequential formats.
  - An abstract interface that defines the application callable
      functions for the class. These are defined in the directory
      `src/<class>/interface`.
  - One or more actual implementations of the classs (for example,
      sparse uniprocessor and parallel matrices implemented with the AIJ storage format).
      These are each in a subdirectory of
      `src/<class>/impls`. Except in rare
      circumstances data
      structures defined here should not be referenced from outside this
      directory.


Each type of object, for instance a vector, is defined in its own
public include file, by
\code
   typedef _p_<class>* <class>;
\endcode
 (for example, `typedef _p_Vec* Vec;`).
  This organization
allows the compiler to perform type checking on all subroutine calls
while at the same time
completely removing the details of the implementation of `_p_<class>` from the application code.
This capability is extremely important
because it allows the library internals to be changed
without altering or recompiling the application code.

Polymorphism is supported through the directory
`src/<class>/interface`,
which contains the code that implements the abstract interface to the
operations on the object.  Essentially, these routines do some error
checking of arguments and logging of profiling information
and then call the function appropriate for the
particular implementation of the object. The name of the abstract
function is `<class>Operation`, for instance, MatMult() or PCCreate(), while
the name of a particular implementation is
`<class>Operation_<implementation>`, for instance,
`MatMult_SeqAIJ()` or `PCCreate_ILU()`. These naming
conventions are used to simplify code maintenance.

\section dev-basic-object-design-common-object-header Common Object Header

All PETSc/PETSc objects have the following common header structures
(in `include/petsc-private/petscimpl.h`)

\code
/* Function table common to all %PETSc compatible classes */
typedef struct {
  int (*getcomm)(PetscObject,MPI_Comm *);
  int (*view)(PetscObject,Viewer);
  int (*destroy)(PetscObject);
  int (*query)(PetscObject,const char *,PetscObject *);
  int (*compose)(PetscObject,const char*,PetscObject);
  int (*composefunction)(PetscObject,const char *,void (*)(void));
  int (*queryfunction)(PetscObject,const char *,void (**)(void));
} PetscOps;

/* Data structure header common to all PETSc compatible classs */

struct _p_<class> {
  PetscClassId     classid;                                  
  PetscOps         *bops;                                   
  <class>Ops        *ops;                                    
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
\endcode
Here `<class>ops` is a function table (like the `PetscOps` above) that
contains the function pointers for the operations specific to that class.
For example, the %PETSc vector class object looks like

\code
/* Function table common to all PETSc compatible vector objects */
typedef struct _VecOps* VecOps;
struct _VecOps {
  PetscErrorCode  (*duplicate)(Vec,Vec*),               /* get single vector               */
       (*duplicatevecs)(Vec,int,Vec**),                 /* get array of vectors            */
       (*destroyvecs)(Vec*,int),                        /* free array of vectors           */
       (*dot)(Vec,Vec,Scalar*),                         /* z = x\^H * y                    */
       (*mdot)(int,Vec,Vec*,Scalar*),                   /* z[j] = x dot y[j]               */
       (*norm)(Vec,NormType,double*),                   /* z = sqrt(x\^H * x)              */
       (*tdot)(Vec,Vec,Scalar*),                        /* x'*y                            */
       (*mtdot)(int,Vec,Vec*,Scalar*),                  /* z[j] = x dot y[j]               */
       (*scale)(Scalar*,Vec),                           /* x = alpha * x                   */
       (*copy)(Vec,Vec),                                /* y = x                           */
       (*set)(Scalar*,Vec),                             /* y = alpha                       */
       (*swap)(Vec,Vec),                                /* exchange x and y                */
       (*axpy)(Scalar*,Vec,Vec),                        /* y = y + alpha * x               */
       (*axpby)(Scalar*,Scalar*,Vec,Vec),               /* y = y + alpha * x + beta * y    */
       (*maxpy)(int,Scalar*,Vec,Vec*),                  /* y = y + alpha[j] x[j]           */
       (*aypx)(Scalar*,Vec,Vec),                        /* y = x + alpha * y               */
       (*waxpy)(Scalar*,Vec,Vec,Vec),                   /* w = y + alpha * x               */
       (*pointwisemult)(Vec,Vec,Vec),                   /* w = x .* y                      */
       (*pointwisedivide)(Vec,Vec,Vec),                 /* w = x ./ y                      */
       (*setvalues)(Vec,int,int*,Scalar*,InsertMode),
       (*assemblybegin)(Vec),                           /* start global assembly           */
       (*assemblyend)(Vec),                             /* end global assembly             */
       (*getarray)(Vec,Scalar**),                       /* get data array                  */
       (*getsize)(Vec,int*),(*getlocalsize)(Vec,int*),
       (*getownershiprange)(Vec,int*,int*),
       (*restorearray)(Vec,Scalar**),                   /* restore data array              */
       (*max)(Vec,int*,double*),                        /* z = max(x); idx=index of max(x) */
       (*min)(Vec,int*,double*),                        /* z = min(x); idx=index of min(x) */
       (*setrandom)(PetscRandom,Vec),                   /* set y[j] = random numbers       */
       (*setoption)(Vec,VecOption),
       (*setvaluesblocked)(Vec,int,int*,Scalar*,InsertMode),
       (*destroy)(Vec),
       (*view)(Vec,Viewer);
};

/* Data structure header common to all %PETSc vector classs */

struct _p_Vec {
  PetscClassId           classid;                                  
  PetscOps               *bops;                                   
  VecOps                 *ops;                                   
  MPI\_Comm               comm;
  PetscLogDouble         flops,time,mem;                          
  int                    id;                                      
  int                    refct;                                   
  int                    tag;                                     
  DLList                 qlist;                                   
  OList                  olist;                                   
  char                   *type_name;                            
  PetscObject            parent;                                  
  char*                  name;                                   
  char                   *prefix;                                
  void**                 fortran_func_pointers;       
  void                   *data;     /* implementation-specific data */
  int                    N, n;      /* global, local vector size    */
  int                    bs;
  ISLocalToGlobalMapping mapping;   /* mapping used in VecSetValuesLocal()        */
  ISLocalToGlobalMapping bmapping;  /* mapping used in VecSetValuesBlockedLocal() */
};
\endcode

Each %PETSc object begins with a PetscClassId which is used for error checking.
Each different class of objects has its value for the classid; these are used
to distinguish between classes. When a new class is created one needs to call
\code
  ierr = PetscClassIdRegister(char *classname,PetscClassId *classid);CHKERRQ(ierr);
\endcode
For example,
\code
  ierr = PetscClassIdRegister(''index set'',&IS_CLASSID);CHKERRQ(ierr);
\endcode
**Question: Why is a fundamental part of %PETSc objects defined in PetscLog when %PETSc Log
is something that can be "turned off"?** One can verify that an object is valid of a particular
class with
\code
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
\endcode
The third argument to this macro indicates the position in the calling sequence of the
function the object was passed in. This is generate more complete error messages.

To check for an object of any type use
\code
  PetscValidHeader(x,1);
\endcode


Several routines are provided for manipulating data within the header,
including
\code
   int PetscObjectGetComm(PetscObject object,MPI_Comm *comm);
\endcode
which returns in `comm` the MPI communicator associated with the
specified object.

\section dev-basic-object-design-common-object-functions Common Object Functions

We now discuss the specific functions in the %PETSc common function table.

  - `getcomm(PetscObject,MPI_Comm *)` obtains the MPI communicator associated
      with this object.

  - `view(PetscObject,Viewer)` allows one to store or visualize the data inside
      an object. If the Viewer is null than should cause the object to print
      information on the object to standard out. %PETSc provides a variety of simple
      viewers.

  - `destroy(PetscObject)` causes the reference count of the object to be decreased
      by one or the object to be destroyed and all memory used by the object to be freed when
      the reference count drops to zero.
      If the object has any other objects composed with it then they are each sent a
      `destroy()`, i.e. the \trl{destroy()} function is called on them also.

  - `compose(PetscObject,char *name, PetscObject)` associates the second object with
      the first object and increases the reference count of the second object. If an
      object with the
      same name was previously composed that object is dereferenced and replaced with
      the new object. If the
      second object is null and and object with the same name has already been
      composed that object is dereferenced (the \trl{destroy()} function is called on
      it, and that object is removed from the first object); i.e. this is a way to
      remove, by name, an object that was previously composed.

  - `query(PetscObject,char *name, PetscObject *)` retrieves an object that was
      previously composed with the first object. Retreives a null if no object with
      that name was previously composed.

  - `composefunction(PetscObject,char *name,char *fname,void *func)` associates a function
      pointer to an object. If the object already had a composed function with the
      same name, the old one is replace. If the fname is null it is removed from
      the object. The string `fname` is the  character string name of the function;
      it may include the path name or URL of the dynamic library where the function is located.
      The argument `name` is a ``short'' name of the function to be used with the
      `queryfunction()` call. On systems that support dynamic libraries the `func`
      argument is ignored; otherwise `func` is the actual function pointer.

      For example, `fname` may be `libpetscksp:PCCreate_LU` or
      %http://www.mcs.anl.gov/petsc/libpetscksp:PCCreate_LU.

  - `queryfunction(PetscObject,char *name,void **func)` retreives a function pointer that
      was associated with the object. If dynamic libraries are used the function is loaded
      into memory at this time (if it has not been previously loaded), not when the
      `composefunction()` routine was called.


Since the object composition allows one to **only** compose %PETSc objects
with %PETSc objects rather than any arbitrary pointer, PETsc provides
the convenience object PetscContainer, created with the
routine PetscContainerCreate(MPI_Comm,PetscContainer)
to allow one to wrap any kind of data into a %PETSc object that can then be
composed with a %PETSc object.

\section dev-basic-object-design-implementation-object-functions PETSc Implementation of the  Object Functions

This sections discusses how %PETSc implements the `compose()`, `query()`, `composefunction()`,
 and `queryfunction()` functions for its object implementations.
Other %PETSc compatible class implementations are free to manage these functions in any
manner; but generally they would use the %PETSc defaults so that the library writer does
not have to ``reinvent the wheel.''

\subsection dev-basic-object-design-compose-and-query Compose and Query

In `src/sys/objects/olist.c` %PETSc defines a C struct
\code
typedef struct _PetscOList *PetscOList;
struct _PetscOList {
    char        name[128];
    PetscObject obj;
    PetscOList  next;
};
\endcode
from which linked lists of composed objects may be constructed. The routines
to manipulate these elementary objects are
\code
  int PetscOListAdd(PetscOList *fl,char *name,PetscObject obj );
  int PetscOListDestroy(PetscOList fl );
  int PetscOListFind(PetscOList fl, char *name, PetscObject *obj)
  int PetscOListDuplicate(PetscOList fl, PetscOList *nl);
\endcode
The function PetscOListAdd() will create the initial PetscOList if the argument
`fl` points to a null.

The %PETSc object `compose()` and `query()` functions are then simply
(defined in `src/sys/objects/inherit.c`)
\code
  PetscErrorCode PetscObjectCompose_Petsc(PetscObject obj,char *name,PetscObject ptr)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscOListAdd(\&obj-$>$olist,name,ptr);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode PetscObjectQuery_Petsc(PetscObject obj,char *name,PetscObject *ptr)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscOListFind(obj-$>$olist,name,ptr);CHKERRQ(ierr);
    PetscFunctionReturn(0); 
  }
\endcode

\subsection dev-basic-object-design-compose-and-query-function Compose and Query Function

%PETSc allows one to compose functions by string name and function pointer. In `src/sys/dll/reg.c`,
%PETSc defines the linked list structure

\code
typedef struct _PetscFunctionList *PetscFunctionList;
struct _PetscFunctionList {
  int  (*routine)(void);
  char *name;
  PetscFunctionList *next;
};
\endcode

Each %PETSc object contains a PetscFunctionList object. The `composefunction()` and
`queryfunction()` are given by

\code
PetscErrorCode PetscObjectComposeFunction_Petsc(PetscObject obj,char *name,void (*ptr)(void))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(\&obj-$>$qlist,name,fname,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectQueryFunction_Petsc(PetscObject obj,char *name,void (**fptr)(void))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListFind(obj-$>$qlist,obj-$>$comm,name,fptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
\endcode

In addition to using the PetscFunctionList mechanism to compose functions into %PETSc objects, it is
also used to allow registration of new class implementations; for example, new
preconditioners, see Section \ref dev-petsc-objects-registering.

\subsection dev-basic-object-design-simple-petsc-objects Simple PETSc Objects

There are some simple %PETSc objects that do not need PETSCHEADER - and
the associated functionality. These objects are internally named as
`_n_<class>` as opposed to `_p_<class>`. For example: `_n_PetscTable`
vs `_p_Vec`.

*/
