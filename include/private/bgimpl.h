#if !defined(_PETSCBGIMPL_H)
#define _PETSCBGIMPL_H

#include <petscbg.h>

typedef struct _n_PetscBGDataLink *PetscBGDataLink;

struct _n_PetscBGDataLink {
  MPI_Datatype atom;
  MPI_Datatype *mine;
  MPI_Datatype *remote;
  PetscBGDataLink next;
};

struct _PetscBGOps {
  int dummy;
};

struct _p_PetscBG {
  PETSCHEADER(struct _PetscBGOps);
  PetscInt        nlocal;       /* Number of local nodes with outgoing edges */
  PetscInt        *mine;        /* Location of nodes with outgoing edges */
  PetscInt        *mine_alloc;
  PetscBGNode     *remote;      /* Remote nodes referenced by outgoing edges */
  PetscBGNode     *remote_alloc;
  PetscInt        nranks;       /* Number of ranks owning nodes addressed by outgoing edges from current rank */
  PetscInt        *ranks;       /* List of ranks referenced by "remote" */
  PetscInt        *counts;      /* Number of remote nodes for each rank */
  PetscBGDataLink link;         /* List of MPI data types and windows, lazily constructed for each data type */
};

#endif
