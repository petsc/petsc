/*
   A star forest (SF) describes a communication pattern
*/
#if !defined(__PETSCSF_H)
#define __PETSCSF_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscClassId PETSCSF_CLASSID;

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
extern const char *const PetscSFSynchronizationTypes[];

#if !defined(PETSC_HAVE_MPI_WIN_CREATE) /* The intent here is to be able to compile even without a complete MPI. */
typedef struct MPI_Win_MISSING *MPI_Win;
#endif

extern PetscErrorCode PetscSFInitializePackage(const char*);
extern PetscErrorCode PetscSFFinalizePackage(void);
extern PetscErrorCode PetscSFCreate(MPI_Comm comm,PetscSF*);
extern PetscErrorCode PetscSFDestroy(PetscSF*);
extern PetscErrorCode PetscSFView(PetscSF,PetscViewer);
extern PetscErrorCode PetscSFSetFromOptions(PetscSF);
extern PetscErrorCode PetscSFSetSynchronizationType(PetscSF,PetscSFSynchronizationType);
extern PetscErrorCode PetscSFSetRankOrder(PetscSF,PetscBool);
extern PetscErrorCode PetscSFSetGraph(PetscSF,PetscInt nroots,PetscInt nleaves,const PetscInt *ilocal,PetscCopyMode modelocal,const PetscSFNode *remote,PetscCopyMode moderemote);
extern PetscErrorCode PetscSFGetGraph(PetscSF,PetscInt *nroots,PetscInt *nleaves,const PetscInt **ilocal,const PetscSFNode **iremote);
extern PetscErrorCode PetscSFCreateEmbeddedSF(PetscSF,PetscInt nroots,const PetscInt *selected,PetscSF *newsf);
extern PetscErrorCode PetscSFCreateArray(PetscSF,MPI_Datatype,void*,void*);
extern PetscErrorCode PetscSFDestroyArray(PetscSF,MPI_Datatype,void*,void*);
extern PetscErrorCode PetscSFReset(PetscSF);
extern PetscErrorCode PetscSFGetRanks(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscMPIInt**,const PetscMPIInt**);
extern PetscErrorCode PetscSFGetDataTypes(PetscSF,MPI_Datatype,const MPI_Datatype**,const MPI_Datatype**);
extern PetscErrorCode PetscSFGetWindow(PetscSF,MPI_Datatype,void*,PetscBool,PetscMPIInt,PetscMPIInt,PetscMPIInt,MPI_Win*);
extern PetscErrorCode PetscSFFindWindow(PetscSF,MPI_Datatype,const void*,MPI_Win*);
extern PetscErrorCode PetscSFRestoreWindow(PetscSF,MPI_Datatype,const void*,PetscBool,PetscMPIInt,MPI_Win*);
extern PetscErrorCode PetscSFGetGroups(PetscSF,MPI_Group*,MPI_Group*);
extern PetscErrorCode PetscSFGetMultiSF(PetscSF,PetscSF*);
extern PetscErrorCode PetscSFCreateInverseSF(PetscSF,PetscSF*);

/* broadcasts rootdata to leafdata */
extern PetscErrorCode PetscSFBcastBegin(PetscSF,MPI_Datatype,const void *rootdata,void *leafdata);
extern PetscErrorCode PetscSFBcastEnd(PetscSF,MPI_Datatype,const void *rootdata,void *leafdata);
/* Reduce leafdata into rootdata using provided operation */
extern PetscErrorCode PetscSFReduceBegin(PetscSF,MPI_Datatype,const void *leafdata,void *rootdata,MPI_Op);
extern PetscErrorCode PetscSFReduceEnd(PetscSF,MPI_Datatype,const void *leafdata,void *rootdata,MPI_Op);
/* Atomically modifies (using provided operation) rootdata using leafdata from each leaf, value at root at time of modification is returned in leafupdate. */
extern PetscErrorCode PetscSFFetchAndOpBegin(PetscSF,MPI_Datatype,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op);
extern PetscErrorCode PetscSFFetchAndOpEnd(PetscSF,MPI_Datatype,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op);
/* Compute the degree of every root vertex (number of leaves in its star) */
extern PetscErrorCode PetscSFComputeDegreeBegin(PetscSF,const PetscInt **degree);
extern PetscErrorCode PetscSFComputeDegreeEnd(PetscSF,const PetscInt **degree);
/* Concatenate data from all leaves into roots */
extern PetscErrorCode PetscSFGatherBegin(PetscSF,MPI_Datatype,const void *leafdata,void *multirootdata);
extern PetscErrorCode PetscSFGatherEnd(PetscSF,MPI_Datatype,const void *leafdata,void *multirootdata);
/* Distribute distinct values to each leaf from roots */
extern PetscErrorCode PetscSFScatterBegin(PetscSF,MPI_Datatype,const void *multirootdata,void *leafdata);
extern PetscErrorCode PetscSFScatterEnd(PetscSF,MPI_Datatype,const void *multirootdata,void *leafdata);

PETSC_EXTERN_CXX_END

#endif
