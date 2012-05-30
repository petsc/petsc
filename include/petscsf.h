/*
   A star forest (SF) describes a communication pattern
*/
#if !defined(__PETSCSF_H)
#define __PETSCSF_H
#include "petscsys.h"

PETSC_EXTERN PetscClassId PETSCSF_CLASSID;

/*S
   PetscSF - PETSc object for communication using star forests

   Level: intermediate

  Concepts: star forest

.seealso: PetscSFCreate()
S*/
typedef struct _p_PetscSF* PetscSF;

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
    PetscSFSynchronizationType - Type of synchronization for PetscSF

$  PETSCSF_SYNCHRONIZATION_FENCE - simplest model, synchronizing across communicator
$  PETSCSF_SYNCHRONIZATION_LOCK - passive model, less synchronous, requires less setup than PETSCSF_SYNCHRONIZATION_ACTIVE, but may require more handshakes
$  PETSCSF_SYNCHRONIZATION_ACTIVE - active model, provides most information to MPI implementation, needs to construct 2-way process groups (more setup than PETSCSF_SYNCHRONIZATION_LOCK)

   Level: beginner

.seealso: PetscSFSetSynchronizationType()
E*/
typedef enum {PETSCSF_SYNCHRONIZATION_FENCE,PETSCSF_SYNCHRONIZATION_LOCK,PETSCSF_SYNCHRONIZATION_ACTIVE} PetscSFSynchronizationType;
PETSC_EXTERN const char *const PetscSFSynchronizationTypes[];

#if !defined(PETSC_HAVE_MPI_WIN_CREATE) /* The intent here is to be able to compile even without a complete MPI. */
typedef struct MPI_Win_MISSING *MPI_Win;
#endif

PETSC_EXTERN PetscErrorCode PetscSFInitializePackage(const char*);
PETSC_EXTERN PetscErrorCode PetscSFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscSFCreate(MPI_Comm comm,PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFDestroy(PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFView(PetscSF,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscSFSetFromOptions(PetscSF);
PETSC_EXTERN PetscErrorCode PetscSFSetSynchronizationType(PetscSF,PetscSFSynchronizationType);
PETSC_EXTERN PetscErrorCode PetscSFSetRankOrder(PetscSF,PetscBool);
PETSC_EXTERN PetscErrorCode PetscSFSetGraph(PetscSF,PetscInt nroots,PetscInt nleaves,const PetscInt *ilocal,PetscCopyMode modelocal,const PetscSFNode *remote,PetscCopyMode moderemote);
PETSC_EXTERN PetscErrorCode PetscSFGetGraph(PetscSF,PetscInt *nroots,PetscInt *nleaves,const PetscInt **ilocal,const PetscSFNode **iremote);
PETSC_EXTERN PetscErrorCode PetscSFCreateEmbeddedSF(PetscSF,PetscInt nroots,const PetscInt *selected,PetscSF *newsf);
PETSC_EXTERN PetscErrorCode PetscSFCreateArray(PetscSF,MPI_Datatype,void*,void*);
PETSC_EXTERN PetscErrorCode PetscSFDestroyArray(PetscSF,MPI_Datatype,void*,void*);
PETSC_EXTERN PetscErrorCode PetscSFReset(PetscSF);
PETSC_EXTERN PetscErrorCode PetscSFGetRanks(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscMPIInt**,const PetscMPIInt**);
PETSC_EXTERN PetscErrorCode PetscSFGetDataTypes(PetscSF,MPI_Datatype,const MPI_Datatype**,const MPI_Datatype**);
PETSC_EXTERN PetscErrorCode PetscSFGetWindow(PetscSF,MPI_Datatype,void*,PetscBool,PetscMPIInt,PetscMPIInt,PetscMPIInt,MPI_Win*);
PETSC_EXTERN PetscErrorCode PetscSFFindWindow(PetscSF,MPI_Datatype,const void*,MPI_Win*);
PETSC_EXTERN PetscErrorCode PetscSFRestoreWindow(PetscSF,MPI_Datatype,const void*,PetscBool,PetscMPIInt,MPI_Win*);
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
