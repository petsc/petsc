#ifndef PARTITIONERIMPL_H
#define PARTITIONERIMPL_H

#include <petscviewer.h>
#include <petscpartitioner.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      PetscPartitionerRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscPartitionerRegisterAll(void);

typedef struct _PetscPartitionerOps *PetscPartitionerOps;
struct _PetscPartitionerOps {
  PetscErrorCode (*setfromoptions)(PetscPartitioner, PetscOptionItems *);
  PetscErrorCode (*setup)(PetscPartitioner);
  PetscErrorCode (*reset)(PetscPartitioner);
  PetscErrorCode (*view)(PetscPartitioner, PetscViewer);
  PetscErrorCode (*destroy)(PetscPartitioner);
  PetscErrorCode (*partition)(PetscPartitioner, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscSection, PetscSection, PetscSection, IS *);
};

struct _p_PetscPartitioner {
  PETSCHEADER(struct _PetscPartitionerOps);
  void       *data;    /* Implementation object */
  PetscInt    height;  /* Height of points to partition into non-overlapping subsets */
  PetscInt    edgeCut; /* The number of edge cut by the partition */
  PetscReal   balance; /* The maximum partition size divided by the minimum size */
  PetscViewer viewer;
  PetscViewer viewerGraph;
  PetscBool   viewGraph;
  PetscBool   noGraph; /* if true, the partitioner does not need the connectivity graph, only the number of local vertices */
  PetscBool   usevwgt; /* if true, the partitioner looks at the local section vertSection to weight the vertices of the graph */
};

#endif
