#pragma once

#include <petscviewer.h>
#include <petscpartitioner.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      PetscPartitionerRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscPartitionerRegisterAll(void);

typedef struct _PetscPartitionerOps *PetscPartitionerOps;
struct _PetscPartitionerOps {
  PetscErrorCode (*setfromoptions)(PetscPartitioner, PetscOptionItems);
  PetscErrorCode (*setup)(PetscPartitioner);
  PetscErrorCode (*reset)(PetscPartitioner);
  PetscErrorCode (*view)(PetscPartitioner, PetscViewer);
  PetscErrorCode (*destroy)(PetscPartitioner);
  PetscErrorCode (*partition)(PetscPartitioner, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscSection, PetscSection, PetscSection, PetscSection, IS *);
};

struct _p_PetscPartitioner {
  PETSCHEADER(struct _PetscPartitionerOps);
  void     *data;    /* Implementation object */
  PetscInt  height;  /* Height of points to partition into non-overlapping subsets */
  PetscInt  edgeCut; /* The number of edge cut by the partition */
  PetscReal balance; /* The maximum partition size divided by the minimum size */

  PetscBool         printHeader;
  PetscViewer       viewer, viewerGraph;
  PetscViewerFormat viewerFmt;

  PetscBool noGraph; /* if true, the partitioner does not need the connectivity graph, only the number of local vertices */
  PetscBool usevwgt; /* if true, the partitioner looks at the local section vertSection to weight the vertices of the graph */
  PetscBool useewgt; /* if true, the partitioner looks at the topology to weight the edges of the graph */
};

/* All levels > 8 logged in 8-th level */
#define PETSCPARTITIONER_MS_MAXSTAGE 8
#define PETSCPARTITIONER_MS_NUMSTAGE (PETSCPARTITIONER_MS_MAXSTAGE + 1)
PETSC_EXTERN PetscLogEvent  PetscPartitioner_MS_SetUp;
PETSC_EXTERN PetscLogEvent  PetscPartitioner_MS_Stage[PETSCPARTITIONER_MS_NUMSTAGE];
PETSC_EXTERN PetscErrorCode PetscPartitionerMultistageGetStages_Multistage(PetscPartitioner, PetscInt *, MPI_Group *[]);
PETSC_EXTERN PetscErrorCode PetscPartitionerMultistageSetStage_Multistage(PetscPartitioner, PetscInt, PetscObject);
PETSC_EXTERN PetscErrorCode PetscPartitionerMultistageGetStage_Multistage(PetscPartitioner, PetscInt *, PetscObject *);
