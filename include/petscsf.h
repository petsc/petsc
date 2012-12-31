/*
   A star forest (SF) describes a communication pattern
*/
#if !defined(__PETSCSF_H)
#define __PETSCSF_H
#include <petscsys.h>

PETSC_EXTERN PetscClassId PETSCSF_CLASSID;

/*S
   PetscSF - PETSc object for communication using star forests

   Level: intermediate

  Concepts: star forest

.seealso: PetscSFCreate()
S*/
typedef struct _p_PetscSF* PetscSF;


/*J
    PetscSFType - String with the name of a PetscSF method or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.so:mysfcreate()

   Level: beginner

.seealso: PetscSFSetType(), PetscSF
J*/
typedef const char *PetscSFType;
#define PETSCSFWINDOW "window"
#define PETSCSFBASIC  "basic"

/*S
   PetscSFNode - specifier of owner and index

   Level: beginner

  Concepts: indexing, stride, distribution

.seealso: PetscSFSetGraph()
S*/
typedef struct {
  PetscInt rank;                /* Rank of owner */
  PetscInt index;               /* Index of node on rank */
} PetscSFNode;

/*E
    PetscSFWindowSyncType - Type of synchronization for PETSCSFWINDOW

$  PETSCSF_WINDOW_SYNC_FENCE - simplest model, synchronizing across communicator
$  PETSCSF_WINDOW_SYNC_LOCK - passive model, less synchronous, requires less setup than PETSCSF_WINDOW_SYNC_ACTIVE, but may require more handshakes
$  PETSCSF_WINDOW_SYNC_ACTIVE - active model, provides most information to MPI implementation, needs to construct 2-way process groups (more setup than PETSCSF_WINDOW_SYNC_LOCK)

   Level: advanced

.seealso: PetscSFWindowSetSyncType(), PetscSFWindowGetSyncType()
E*/
typedef enum {PETSCSF_WINDOW_SYNC_FENCE,PETSCSF_WINDOW_SYNC_LOCK,PETSCSF_WINDOW_SYNC_ACTIVE} PetscSFWindowSyncType;
PETSC_EXTERN const char *const PetscSFWindowSyncTypes[];

/*E
    PetscSFDuplicateOption - Aspects to preserve when duplicating a PetscSF

$  PETSCSF_DUPLICATE_CONFONLY - configuration only, user must call PetscSFSetGraph()
$  PETSCSF_DUPLICATE_RANKS - communication ranks preserved, but different graph (allows simpler setup after calling PetscSFSetGraph())
$  PETSCSF_DUPLICATE_GRAPH - entire graph duplicated

   Level: beginner

.seealso: PetscSFDuplicate()
E*/
typedef enum {PETSCSF_DUPLICATE_CONFONLY,PETSCSF_DUPLICATE_RANKS,PETSCSF_DUPLICATE_GRAPH} PetscSFDuplicateOption;
PETSC_EXTERN const char *const PetscSFDuplicateOptions[];

PETSC_EXTERN PetscFList PetscSFList;
PETSC_EXTERN PetscErrorCode PetscSFRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode PetscSFRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode PetscSFRegister(const char[],const char[],const char[],PetscErrorCode (*)(PetscSF));

/*MC
   PetscSFRegisterDynamic - Adds an implementation of the PetscSF communication protocol.

   Synopsis:
   PetscErrorCode PetscSFRegisterDynamic(const char *name_method,const char *path,const char *name_create,PetscErrorCode (*routine_create)(PetscSF))

   Not collective

   Input Parameters:
+  name_impl - name of a new user-defined implementation
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   PetscSFRegisterDynamic() may be called multiple times to add several user-defined implementations.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
   and others of the form ${any_environmental_variable} occuring in pathname will be
   replaced with appropriate values.

   Sample usage:
.vb
   PetscSFRegisterDynamic("my_impl",/home/username/my_lib/lib/libg/solaris/mylib.a,
                "MyImplCreate",MyImplCreate);
.ve

   Then, this implementation can be chosen with the procedural interface via
$     PetscSFSetType(sf,"my_impl")
   or at runtime via the option
$     -snes_type my_solver

   Level: advanced

    Note: If your function is not being put into a shared library then use PetscSFRegister() instead

.keywords: PetscSF, register

.seealso: PetscSFRegisterAll(), PetscSFRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscSFRegisterDynamic(a,b,c,d) PetscSFRegister(a,b,c,0)
#else
#define PetscSFRegisterDynamic(a,b,c,d) PetscSFRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode PetscSFInitializePackage(const char*);
PETSC_EXTERN PetscErrorCode PetscSFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscSFCreate(MPI_Comm comm,PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFDestroy(PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFSetType(PetscSF,PetscSFType);
PETSC_EXTERN PetscErrorCode PetscSFView(PetscSF,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscSFSetUp(PetscSF);
PETSC_EXTERN PetscErrorCode PetscSFSetFromOptions(PetscSF);
PETSC_EXTERN PetscErrorCode PetscSFDuplicate(PetscSF,PetscSFDuplicateOption,PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFWindowSetSyncType(PetscSF,PetscSFWindowSyncType);
PETSC_EXTERN PetscErrorCode PetscSFWindowGetSyncType(PetscSF,PetscSFWindowSyncType*);
PETSC_EXTERN PetscErrorCode PetscSFSetRankOrder(PetscSF,PetscBool);
PETSC_EXTERN PetscErrorCode PetscSFSetGraph(PetscSF,PetscInt,PetscInt,const PetscInt*,PetscCopyMode,const PetscSFNode*,PetscCopyMode);
PETSC_EXTERN PetscErrorCode PetscSFGetGraph(PetscSF,PetscInt *nroots,PetscInt *nleaves,const PetscInt **ilocal,const PetscSFNode **iremote);
PETSC_EXTERN PetscErrorCode PetscSFGetLeafRange(PetscSF,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSFCreateEmbeddedSF(PetscSF,PetscInt nroots,const PetscInt *selected,PetscSF *newsf);
PETSC_EXTERN PetscErrorCode PetscSFCreateArray(PetscSF,MPI_Datatype,void*,void*);
PETSC_EXTERN PetscErrorCode PetscSFDestroyArray(PetscSF,MPI_Datatype,void*,void*);
PETSC_EXTERN PetscErrorCode PetscSFReset(PetscSF);
PETSC_EXTERN PetscErrorCode PetscSFGetRanks(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**,const PetscInt**);
PETSC_EXTERN PetscErrorCode PetscSFGetGroups(PetscSF,MPI_Group*,MPI_Group*);
PETSC_EXTERN PetscErrorCode PetscSFGetMultiSF(PetscSF,PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFCreateInverseSF(PetscSF,PetscSF*);

/* broadcasts rootdata to leafdata */
PETSC_EXTERN PetscErrorCode PetscSFBcastBegin(PetscSF,MPI_Datatype,const void *rootdata,void *leafdata);
PETSC_EXTERN PetscErrorCode PetscSFBcastEnd(PetscSF,MPI_Datatype,const void *rootdata,void *leafdata);
/* Reduce leafdata into rootdata using provided operation */
PETSC_EXTERN PetscErrorCode PetscSFReduceBegin(PetscSF,MPI_Datatype,const void *leafdata,void *rootdata,MPI_Op);
PETSC_EXTERN PetscErrorCode PetscSFReduceEnd(PetscSF,MPI_Datatype,const void *leafdata,void *rootdata,MPI_Op);
/* Atomically modifies (using provided operation) rootdata using leafdata from each leaf, value at root at time of modification is returned in leafupdate. */
PETSC_EXTERN PetscErrorCode PetscSFFetchAndOpBegin(PetscSF,MPI_Datatype,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op);
PETSC_EXTERN PetscErrorCode PetscSFFetchAndOpEnd(PetscSF,MPI_Datatype,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op);
/* Compute the degree of every root vertex (number of leaves in its star) */
PETSC_EXTERN PetscErrorCode PetscSFComputeDegreeBegin(PetscSF,const PetscInt **degree);
PETSC_EXTERN PetscErrorCode PetscSFComputeDegreeEnd(PetscSF,const PetscInt **degree);
/* Concatenate data from all leaves into roots */
PETSC_EXTERN PetscErrorCode PetscSFGatherBegin(PetscSF,MPI_Datatype,const void *leafdata,void *multirootdata);
PETSC_EXTERN PetscErrorCode PetscSFGatherEnd(PetscSF,MPI_Datatype,const void *leafdata,void *multirootdata);
/* Distribute distinct values to each leaf from roots */
PETSC_EXTERN PetscErrorCode PetscSFScatterBegin(PetscSF,MPI_Datatype,const void *multirootdata,void *leafdata);
PETSC_EXTERN PetscErrorCode PetscSFScatterEnd(PetscSF,MPI_Datatype,const void *multirootdata,void *leafdata);


#endif
