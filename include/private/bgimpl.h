#if !defined(_PETSCBGIMPL_H)
#define _PETSCBGIMPL_H

#include <petscbg.h>

typedef struct _n_PetscBGDataLink *PetscBGDataLink;
typedef struct _n_PetscBGWinLink  *PetscBGWinLink;

struct _n_PetscBGDataLink {
  MPI_Datatype unit;
  MPI_Datatype *mine;
  MPI_Datatype *remote;
  PetscBGDataLink next;
};

struct _n_PetscBGWinLink {
  size_t         bytes;
  void           *addr;
  MPI_Win        win;
  PetscBGWinLink next;
};

struct _PetscBGOps {
  int dummy;
};

struct _p_PetscBG {
  PETSCHEADER(struct _PetscBGOps);
  PetscInt        nowned;       /* Number of owned nodes (candidates for incoming edges) */
  PetscInt        nlocal;       /* Number of local nodes with outgoing edges */
  PetscInt        *mine;        /* Location of nodes with outgoing edges */
  PetscInt        *mine_alloc;
  PetscBGNode     *remote;      /* Remote nodes referenced by outgoing edges */
  PetscBGNode     *remote_alloc;
  PetscInt        nranks;       /* Number of ranks owning nodes addressed by outgoing edges from current rank */
  PetscInt        *ranks;       /* List of ranks referenced by "remote" */
  PetscInt        *roffset;     /* Array of length nranks+1, offset in rmine/rremote for each rank */
  PetscMPIInt     *rmine;       /* Concatenated array holding local indices referencing each remote rank */
  PetscMPIInt     *rremote;     /* Concatenated array holding remote indices referenced for each remote rank */
  PetscBGDataLink link;         /* List of MPI data types and windows, lazily constructed for each data type */
  PetscBGWinLink  wins;         /* List of active windows */
};

#endif
