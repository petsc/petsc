/*
    Index sets for scatter-gather type operations in vectors
and matrices.

*/

#if !defined(_IS_H)
#define _IS_H

#include <petscis.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool ISRegisterAllCalled;
PETSC_EXTERN PetscBool ISLocalToGlobalMappingRegisterAllCalled;
PETSC_EXTERN PetscErrorCode ISRegisterAll(void);

struct _ISOps {
  PetscErrorCode (*getindices)(IS,const PetscInt*[]);
  PetscErrorCode (*restoreindices)(IS,const PetscInt*[]);
  PetscErrorCode (*invertpermutation)(IS,PetscInt,IS*);
  PetscErrorCode (*sort)(IS);
  PetscErrorCode (*sortremovedups)(IS);
  PetscErrorCode (*sorted)(IS,PetscBool*);
  PetscErrorCode (*duplicate)(IS,IS*);
  PetscErrorCode (*destroy)(IS);
  PetscErrorCode (*view)(IS,PetscViewer);
  PetscErrorCode (*load)(IS,PetscViewer);
  PetscErrorCode (*identity)(IS,PetscBool*);
  PetscErrorCode (*copy)(IS,IS);
  PetscErrorCode (*togeneral)(IS);
  PetscErrorCode (*oncomm)(IS,MPI_Comm,PetscCopyMode,IS*);
  PetscErrorCode (*setblocksize)(IS,PetscInt);
  PetscErrorCode (*contiguous)(IS,PetscInt,PetscInt,PetscInt*,PetscBool*);
  PetscErrorCode (*locate)(IS,PetscInt,PetscInt *);
};

struct _p_IS {
  PETSCHEADER(struct _ISOps);
  PetscLayout  map;
  PetscBool    isperm;          /* if is a permutation */
  PetscInt     max,min;         /* range of possible values */
  void         *data;
  PetscBool    isidentity;
  PetscInt     *total, *nonlocal;   /* local representation of ALL indices across the comm as well as the nonlocal part. */
  PetscInt     local_offset;        /* offset to the local part within the total index set */
  IS           complement;          /* IS wrapping nonlocal indices. */
};

extern PetscErrorCode ISLoad_Default(IS, PetscViewer);

struct _ISLocalToGlobalMappingOps {
  PetscErrorCode (*globaltolocalmappingsetup)(ISLocalToGlobalMapping);
  PetscErrorCode (*globaltolocalmappingapply)(ISLocalToGlobalMapping,ISGlobalToLocalMappingMode,PetscInt,const PetscInt[],PetscInt*,PetscInt[]);
  PetscErrorCode (*globaltolocalmappingapplyblock)(ISLocalToGlobalMapping,ISGlobalToLocalMappingMode,PetscInt,const PetscInt[],PetscInt*,PetscInt[]);
  PetscErrorCode (*destroy)(ISLocalToGlobalMapping);
};

struct _p_ISLocalToGlobalMapping{
  PETSCHEADER(struct _ISLocalToGlobalMappingOps);
  PetscInt     n;               /* number of local indices */
  PetscInt     bs;              /* blocksize; there is one index per block */
  PetscInt    *indices;         /* global index of each local index */
  PetscInt     globalstart;     /* first global referenced in indices */
  PetscInt     globalend;       /* last + 1 global referenced in indices */
  PetscBool    info_cached;     /* reuse GetInfo */
  PetscBool    info_free;
  PetscInt     info_nproc;
  PetscInt    *info_procs;
  PetscInt    *info_numprocs;
  PetscInt   **info_indices;
  PetscInt    *info_nodec;
  PetscInt   **info_nodei;
  void        *data;            /* type specific data is stored here */
};

struct _n_ISColoring {
  PetscInt        refct;
  PetscInt        n;                /* number of colors */
  IS              *is;              /* for each color indicates columns */
  MPI_Comm        comm;
  ISColoringValue *colors;          /* for each column indicates color */
  PetscInt        N;                /* number of columns */
  ISColoringType  ctype;
  PetscBool       allocated;
};

#endif
