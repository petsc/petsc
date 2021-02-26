/*
   A star forest (SF) describes a communication pattern
*/
#if !defined(PETSCSF_H)
#define PETSCSF_H
#include <petscsys.h>
#include <petscsftypes.h>
#include <petscvec.h> /* for Vec, VecScatter etc */

PETSC_EXTERN PetscClassId PETSCSF_CLASSID;

#define PETSCSFBASIC      "basic"
#define PETSCSFNEIGHBOR   "neighbor"
#define PETSCSFALLGATHERV "allgatherv"
#define PETSCSFALLGATHER  "allgather"
#define PETSCSFGATHERV    "gatherv"
#define PETSCSFGATHER     "gather"
#define PETSCSFALLTOALL   "alltoall"
#define PETSCSFWINDOW     "window"

/*E
   PetscSFPattern - Pattern of the PetscSF graph

$  PETSCSF_PATTERN_GENERAL   - A general graph. One sets the graph with PetscSFSetGraph() and usually does not use this enum directly.
$  PETSCSF_PATTERN_ALLGATHER - A graph that every rank gathers all roots from all ranks (like MPI_Allgather/v). One sets the graph with PetscSFSetGraphWithPattern().
$  PETSCSF_PATTERN_GATHER    - A graph that rank 0 gathers all roots from all ranks (like MPI_Gather/v with root=0). One sets the graph with PetscSFSetGraphWithPattern().
$  PETSCSF_PATTERN_ALLTOALL  - A graph that every rank gathers different roots from all ranks (like MPI_Alltoall). One sets the graph with PetscSFSetGraphWithPattern().
                               In an ALLTOALL graph, we assume each process has <size> leaves and <size> roots, with each leaf connecting to a remote root. Here <size> is
                               the size of the communicator. This does not mean one can not communicate multiple data items between a pair of processes. One just needs to
                               create a new MPI datatype for the multiple data items, e.g., by MPI_Type_contiguous.
   Level: beginner

.seealso: PetscSFSetGraph(), PetscSFSetGraphWithPattern()
E*/
typedef enum {PETSCSF_PATTERN_GENERAL=0,PETSCSF_PATTERN_ALLGATHER,PETSCSF_PATTERN_GATHER,PETSCSF_PATTERN_ALLTOALL} PetscSFPattern;

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
    PetscSFWindowFlavorType - Flavor for the creation of MPI windows for PETSCSFWINDOW

$  PETSCSF_WINDOW_FLAVOR_CREATE - Use MPI_Win_create, no reusage
$  PETSCSF_WINDOW_FLAVOR_DYNAMIC - Use MPI_Win_create_dynamic and dynamically attach pointers
$  PETSCSF_WINDOW_FLAVOR_ALLOCATE - Use MPI_Win_allocate
$  PETSCSF_WINDOW_FLAVOR_SHARED - Use MPI_Win_allocate_shared

   Level: advanced

.seealso: PetscSFWindowSetFlavorType(), PetscSFWindowGetFlavorType()
E*/
typedef enum {PETSCSF_WINDOW_FLAVOR_CREATE,PETSCSF_WINDOW_FLAVOR_DYNAMIC,PETSCSF_WINDOW_FLAVOR_ALLOCATE,PETSCSF_WINDOW_FLAVOR_SHARED} PetscSFWindowFlavorType;
PETSC_EXTERN const char *const PetscSFWindowFlavorTypes[];

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

PETSC_EXTERN PetscFunctionList PetscSFList;
PETSC_EXTERN PetscErrorCode PetscSFRegister(const char[],PetscErrorCode (*)(PetscSF));

PETSC_EXTERN PetscErrorCode PetscSFInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscSFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscSFCreate(MPI_Comm,PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFDestroy(PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFSetType(PetscSF,PetscSFType);
PETSC_EXTERN PetscErrorCode PetscSFGetType(PetscSF,PetscSFType*);
PETSC_EXTERN PetscErrorCode PetscSFView(PetscSF,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscSFViewFromOptions(PetscSF,PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode PetscSFSetUp(PetscSF);
PETSC_EXTERN PetscErrorCode PetscSFSetFromOptions(PetscSF);
PETSC_EXTERN PetscErrorCode PetscSFDuplicate(PetscSF,PetscSFDuplicateOption,PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFWindowSetSyncType(PetscSF,PetscSFWindowSyncType);
PETSC_EXTERN PetscErrorCode PetscSFWindowGetSyncType(PetscSF,PetscSFWindowSyncType*);
PETSC_EXTERN PetscErrorCode PetscSFWindowSetFlavorType(PetscSF,PetscSFWindowFlavorType);
PETSC_EXTERN PetscErrorCode PetscSFWindowGetFlavorType(PetscSF,PetscSFWindowFlavorType*);
PETSC_EXTERN PetscErrorCode PetscSFWindowSetInfo(PetscSF,MPI_Info);
PETSC_EXTERN PetscErrorCode PetscSFWindowGetInfo(PetscSF,MPI_Info*);
PETSC_EXTERN PetscErrorCode PetscSFSetRankOrder(PetscSF,PetscBool);
PETSC_EXTERN PetscErrorCode PetscSFSetGraph(PetscSF,PetscInt,PetscInt,const PetscInt*,PetscCopyMode,const PetscSFNode*,PetscCopyMode);
PETSC_EXTERN PetscErrorCode PetscSFSetGraphWithPattern(PetscSF,PetscLayout,PetscSFPattern);
PETSC_EXTERN PetscErrorCode PetscSFGetGraph(PetscSF,PetscInt*,PetscInt*,const PetscInt**,const PetscSFNode**);
PETSC_EXTERN PetscErrorCode PetscSFGetLeafRange(PetscSF,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSFCreateEmbeddedSF(PetscSF,PetscInt,const PetscInt*,PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFCreateEmbeddedLeafSF(PetscSF,PetscInt,const PetscInt *, PetscSF *);
PETSC_EXTERN PetscErrorCode PetscSFReset(PetscSF);
PETSC_EXTERN PetscErrorCode PetscSFSetUpRanks(PetscSF,MPI_Group);
PETSC_EXTERN PetscErrorCode PetscSFGetRootRanks(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**,const PetscInt**);
PETSC_EXTERN PetscErrorCode PetscSFGetLeafRanks(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**);
PETSC_EXTERN PetscErrorCode PetscSFGetGroups(PetscSF,MPI_Group*,MPI_Group*);
PETSC_EXTERN PetscErrorCode PetscSFGetMultiSF(PetscSF,PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFCreateInverseSF(PetscSF,PetscSF*);

/* Build PetscSF from PetscLayout */
PETSC_EXTERN PetscErrorCode PetscSFSetGraphLayout(PetscSF,PetscLayout,PetscInt,const PetscInt*,PetscCopyMode,const PetscInt*);
PETSC_EXTERN PetscErrorCode PetscSFCreateFromLayouts(PetscLayout,PetscLayout,PetscSF*);
PETSC_DEPRECATED_FUNCTION("Use PetscSFCreateFromLayouts (since v3.15)")
PETSC_STATIC_INLINE PetscErrorCode PetscLayoutsCreateSF(PetscLayout rmap, PetscLayout lmap, PetscSF* sf) { return PetscSFCreateFromLayouts(rmap, lmap, sf); }

/* PetscSection interoperability */
PETSC_EXTERN PetscErrorCode PetscSFSetGraphSection(PetscSF,PetscSection,PetscSection);
PETSC_EXTERN PetscErrorCode PetscSFCreateRemoteOffsets(PetscSF, PetscSection, PetscSection, PetscInt **);
PETSC_EXTERN PetscErrorCode PetscSFDistributeSection(PetscSF, PetscSection, PetscInt **, PetscSection);
PETSC_EXTERN PetscErrorCode PetscSFCreateSectionSF(PetscSF, PetscSection, PetscInt [], PetscSection, PetscSF *);

/* Reduce rootdata to leafdata using provided operation */
PETSC_EXTERN PetscErrorCode PetscSFBcastAndOpBegin(PetscSF,MPI_Datatype,const void*,void*,MPI_Op)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2);
PETSC_EXTERN PetscErrorCode PetscSFBcastAndOpEnd(PetscSF,MPI_Datatype,const void*,void*,MPI_Op)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2);
PETSC_EXTERN PetscErrorCode PetscSFBcastAndOpWithMemTypeBegin(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void*,MPI_Op)
  PetscAttrMPIPointerWithType(4,2) PetscAttrMPIPointerWithType(6,2);

/* Reduce leafdata into rootdata using provided operation */
PETSC_EXTERN PetscErrorCode PetscSFReduceBegin(PetscSF,MPI_Datatype,const void*,void *,MPI_Op)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2);
PETSC_EXTERN PetscErrorCode PetscSFReduceEnd(PetscSF,MPI_Datatype,const void*,void*,MPI_Op)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2);
PETSC_EXTERN PetscErrorCode PetscSFReduceWithMemTypeBegin(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void *,MPI_Op)
  PetscAttrMPIPointerWithType(4,2) PetscAttrMPIPointerWithType(6,2);
/* Atomically modifies (using provided operation) rootdata using leafdata from each leaf, value at root at time of modification is returned in leafupdate. */
PETSC_EXTERN PetscErrorCode PetscSFFetchAndOpBegin(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2) PetscAttrMPIPointerWithType(5,2);
PETSC_EXTERN PetscErrorCode PetscSFFetchAndOpEnd(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2) PetscAttrMPIPointerWithType(5,2);
/* Compute the degree of every root vertex (number of leaves in its star) */
PETSC_EXTERN PetscErrorCode PetscSFComputeDegreeBegin(PetscSF,const PetscInt**);
PETSC_EXTERN PetscErrorCode PetscSFComputeDegreeEnd(PetscSF,const PetscInt**);
PETSC_EXTERN PetscErrorCode PetscSFComputeMultiRootOriginalNumbering(PetscSF,const PetscInt[],PetscInt*,PetscInt*[]);
/* Concatenate data from all leaves into roots */
PETSC_EXTERN PetscErrorCode PetscSFGatherBegin(PetscSF,MPI_Datatype,const void*,void*)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2);
PETSC_EXTERN PetscErrorCode PetscSFGatherEnd(PetscSF,MPI_Datatype,const void*,void*)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2);
/* Distribute distinct values to each leaf from roots */
PETSC_EXTERN PetscErrorCode PetscSFScatterBegin(PetscSF,MPI_Datatype,const void*,void*)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2);
PETSC_EXTERN PetscErrorCode PetscSFScatterEnd(PetscSF,MPI_Datatype,const void*,void*)
  PetscAttrMPIPointerWithType(3,2) PetscAttrMPIPointerWithType(4,2);

PETSC_EXTERN PetscErrorCode PetscSFCompose(PetscSF,PetscSF,PetscSF*);
PETSC_EXTERN PetscErrorCode PetscSFComposeInverse(PetscSF,PetscSF,PetscSF*);

#if defined(MPI_REPLACE)
#  define MPIU_REPLACE MPI_REPLACE
#else
/* When using an old MPI such that MPI_REPLACE is not defined, we do not pass MPI_REPLACE to MPI at all.  Instead, we
 * use it as a flag for our own reducer in the PETSCSFBASIC implementation.  This could be any unique value unlikely to
 * collide with another MPI_Op so we'll just use the value that has been used by every version of MPICH since
 * MPICH2-1.0.6. */
#  define MPIU_REPLACE (MPI_Op)(0x5800000d)
#endif

PETSC_DEPRECATED_FUNCTION("Use PetscSFGetRootRanks (since v3.12)")
PETSC_STATIC_INLINE PetscErrorCode PetscSFGetRanks(PetscSF sf,PetscInt *nranks,const PetscMPIInt **ranks,const PetscInt **roffset,const PetscInt **rmine,const PetscInt **rremote)
{ return PetscSFGetRootRanks(sf,nranks,ranks,roffset,rmine,rremote); }

/*@C
   PetscSFBcastBegin - begin pointwise broadcast to be concluded with call to PetscSFBcastEnd()

   Collective on PetscSF

   Input Arguments:
+  sf - star forest on which to communicate
.  unit - data type associated with each node
-  rootdata - buffer to broadcast

   Output Arguments:
.  leafdata - buffer to update with values from each leaf's respective root

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFSetGraph(), PetscSFView(), PetscSFBcastEnd(), PetscSFReduceBegin(), PetscSFBcastAndOpBegin()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscSFBcastBegin(PetscSF sf,MPI_Datatype unit,const void* rootdata,void* leafdata)
{ return PetscSFBcastAndOpBegin(sf,unit,rootdata,leafdata,MPIU_REPLACE); }

PETSC_STATIC_INLINE PetscErrorCode PetscSFBcastWithMemTypeBegin(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void* rootdata,PetscMemType leafmtype,void* leafdata)
{ return PetscSFBcastAndOpWithMemTypeBegin(sf,unit,rootmtype,rootdata,leafmtype,leafdata,MPIU_REPLACE); }

/*@C
   PetscSFBcastEnd - end a broadcast operation started with PetscSFBcastBegin()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  rootdata - buffer to broadcast

   Output Arguments:
.  leafdata - buffer to update with values from each leaf's respective root

   Level: intermediate

.seealso: PetscSFSetGraph(), PetscSFReduceEnd()
@*/
PETSC_STATIC_INLINE PetscErrorCode PetscSFBcastEnd(PetscSF sf,MPI_Datatype unit,const void* rootdata,void* leafdata)
{ return PetscSFBcastAndOpEnd(sf,unit,rootdata,leafdata,MPIU_REPLACE); }

#endif
