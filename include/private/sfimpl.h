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
  PetscInt        nroots;       /* Number of root vertices on current process (candidates for incoming edges) */
  PetscInt        nleaves;      /* Number of leaf vertices on current process (this process specifies a root for each leaf) */
  PetscInt        *mine;        /* Location of leaves in leafdata arrays provided to the communication routines */
  PetscInt        *mine_alloc;
  PetscSFNode     *remote;      /* Remote references to roots for each local leaf */
  PetscSFNode     *remote_alloc;
  PetscInt        nranks;       /* Number of ranks owning roots connected to my leaves */
  PetscMPIInt     *ranks;       /* List of ranks referenced by "remote" */
  PetscInt        *roffset;     /* Array of length nranks+1, offset in rmine/rremote for each rank */
  PetscMPIInt     *rmine;       /* Concatenated array holding local indices referencing each remote rank */
  PetscMPIInt     *rremote;     /* Concatenated array holding remote indices referenced for each remote rank */
  PetscSFDataLink link;         /* List of MPI data types and windows, lazily constructed for each data type */
  PetscSFWinLink  wins;         /* List of active windows */
  PetscBool       degreeknown;  /* The degree is currently known, do not have to recompute */
  PetscInt        *degree;      /* Degree of each of my root vertices */
  PetscInt        *degreetmp;   /* Temporary local array for computing degree */
  PetscSFSynchronizationType sync; /* FENCE, LOCK, or ACTIVE synchronization */
  PetscBool       rankorder;    /* Sort ranks for gather and scatter operations */
  MPI_Group       ingroup;      /* Group of processes connected to my roots */
  MPI_Group       outgroup;     /* Group of processes connected to my leaves */
  PetscSF         multi;        /* Internal graph used to implement gather and scatter operations */
  PetscBool       graphset;     /* Flag indicating that the graph has been set, required before calling communication routines */
};

#endif
