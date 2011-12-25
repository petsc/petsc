#if !defined(_PETSCSFIMPL_H)
#define _PETSCSFIMPL_H

#include <petscsf.h>

typedef struct _n_PetscSFDataLink *PetscSFDataLink;
typedef struct _n_PetscSFWinLink  *PetscSFWinLink;

struct _n_PetscSFDataLink {
  MPI_Datatype unit;
  MPI_Datatype *mine;
  MPI_Datatype *remote;
  PetscSFDataLink next;
};

struct _n_PetscSFWinLink {
  PetscBool      inuse;
  size_t         bytes;
  void           *addr;
  MPI_Win        win;
  PetscBool      epoch;
  PetscSFWinLink next;
};

struct _PetscSFOps {
  int dummy;
};

struct _p_PetscSF {
  PETSCHEADER(struct _PetscSFOps);
  PetscInt        nowned;       /* Number of owned nodes (candidates for incoming edges) */
  PetscInt        nlocal;       /* Number of local nodes with outgoing edges */
  PetscInt        *mine;        /* Location of nodes with outgoing edges */
  PetscInt        *mine_alloc;
  PetscSFNode     *remote;      /* Remote nodes referenced by outgoing edges */
  PetscSFNode     *remote_alloc;
  PetscInt        nranks;       /* Number of ranks owning nodes addressed by outgoing edges from current rank */
  PetscInt        *ranks;       /* List of ranks referenced by "remote" */
  PetscInt        *roffset;     /* Array of length nranks+1, offset in rmine/rremote for each rank */
  PetscMPIInt     *rmine;       /* Concatenated array holding local indices referencing each remote rank */
  PetscMPIInt     *rremote;     /* Concatenated array holding remote indices referenced for each remote rank */
  PetscSFDataLink link;         /* List of MPI data types and windows, lazily constructed for each data type */
  PetscSFWinLink  wins;         /* List of active windows */
  PetscInt        *degree;      /* Degree of each owned vertex */
  PetscInt        *degreetmp;   /* Temporary local array for computing degree */
  PetscSFSynchronizationType sync; /* FENCE, LOCK, or ACTIVE synchronization */
  PetscBool       rankorder;    /* Sort ranks for gather and scatter operations */
  MPI_Group       ingroup;
  MPI_Group       outgroup;
  PetscSF         multi;        /* Internal graph used to implement gather and scatter operations */
};

#endif
