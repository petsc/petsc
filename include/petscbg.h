/*
   A bipartite graph describes a general communication strategy.
*/
#if !defined(__PETSCBG_H)
#define __PETSCBG_H
#include "petscis.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscClassId PETSCBG_CLASSID;

/*S
   PetscBG - PETSc object for communication using a bipartite graph

   Level: intermediate

  Concepts: bipartite graph

.seealso: PetscBGCreate()
S*/
typedef struct _p_PetscBG* PetscBG;

/*S
   PetscBGNode - specifier of owner and index

   Level: beginner

  Concepts: indexing, stride, distribution

.seealso: PetscBGSetGraph()
S*/
typedef struct {
  PetscInt rank;                /* Rank of owner */
  PetscInt index;               /* Index of node on rank */
} PetscBGNode;

extern PetscErrorCode PetscBGInitializePackage(const char*);
extern PetscErrorCode PetscBGFinalizePackage(void);
extern PetscErrorCode PetscBGCreate(MPI_Comm comm,PetscBG*);
extern PetscErrorCode PetscBGDestroy(PetscBG*);
extern PetscErrorCode PetscBGView(PetscBG,PetscViewer);
extern PetscErrorCode PetscBGSetGraph(PetscBG,PetscInt nowned,PetscInt nlocal,const PetscInt *ilocal,PetscCopyMode modelocal,const PetscBGNode *remote,PetscCopyMode moderemote);
extern PetscErrorCode PetscBGCreateArray(PetscBG,MPI_Datatype,void*,void*);
extern PetscErrorCode PetscBGDestroyArray(PetscBG,MPI_Datatype,void*,void*);
extern PetscErrorCode PetscBGReset(PetscBG);
extern PetscErrorCode PetscBGGetRanks(PetscBG,PetscInt*,const PetscInt**,const PetscInt**,const PetscMPIInt**,const PetscMPIInt**);
extern PetscErrorCode PetscBGGetDataTypes(PetscBG,MPI_Datatype,const MPI_Datatype**,const MPI_Datatype**);
extern PetscErrorCode PetscBGGetWindow(PetscBG,MPI_Datatype,void*,MPI_Win*);
extern PetscErrorCode PetscBGFindWindow(PetscBG,MPI_Datatype,const void*,MPI_Win*);
extern PetscErrorCode PetscBGRestoreWindow(PetscBG,MPI_Datatype,const void*,MPI_Win*);

/* Provide an owned buffer, updates ghosted space */
extern PetscErrorCode PetscBGBcastBegin(PetscBG,MPI_Datatype,const void *owned,void *ghosted);
extern PetscErrorCode PetscBGBcastEnd(PetscBG,MPI_Datatype,const void *owned,void *ghosted);
/* Reduce all ghosted copies to owned space */
extern PetscErrorCode PetscBGReduceBegin(PetscBG,MPI_Datatype,const void *ghosted,void *owned,MPI_Op);
extern PetscErrorCode PetscBGReduceEnd(PetscBG,MPI_Datatype,const void *ghosted,void *owned,MPI_Op);
/* Compute the degree of every owned point */
extern PetscErrorCode PetscBGComputeDegreeBegin(PetscBG,const PetscInt **degree);
extern PetscErrorCode PetscBGComputeDegreeEnd(PetscBG,const PetscInt **degree);
/* Concatenate data from all ghosted copies of each point, owned data first at each entry */
extern PetscErrorCode PetscBGGatherBegin(PetscBG,MPI_Datatype,const void *ghosted,void *multi);
extern PetscErrorCode PetscBGGatherEnd(PetscBG,MPI_Datatype,const void *ghosted,void *multi);
/* Distribute separate values for each point to each ghosted point */
extern PetscErrorCode PetscBGScatterBegin(PetscBG,MPI_Datatype,const void *multi,void *ghosted);
extern PetscErrorCode PetscBGScatterEnd(PetscBG,MPI_Datatype,const void *multi,void *ghosted);
/* Provide owned values, ghosted values sent to their owners and combined with op, values prior to op returned in result */
extern PetscErrorCode PetscBGFetchAndOpBegin(PetscBG,MPI_Datatype,void *owned,const void *ghosted,void *result,MPI_Op);
extern PetscErrorCode PetscBGFetchAndOpEnd(PetscBG,MPI_Datatype,void *owned,const void *ghosted,void *result,MPI_Op);

PETSC_EXTERN_CXX_END

#endif
