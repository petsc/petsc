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

/* events */
PETSC_EXTERN PetscLogEvent IS_View;
PETSC_EXTERN PetscLogEvent IS_Load;

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
  PetscErrorCode (*copy)(IS,IS);
  PetscErrorCode (*togeneral)(IS);
  PetscErrorCode (*oncomm)(IS,MPI_Comm,PetscCopyMode,IS*);
  PetscErrorCode (*setblocksize)(IS,PetscInt);
  PetscErrorCode (*contiguous)(IS,PetscInt,PetscInt,PetscInt*,PetscBool*);
  PetscErrorCode (*locate)(IS,PetscInt,PetscInt *);
  PetscErrorCode (*sortedlocal)(IS,PetscBool*);
  PetscErrorCode (*sortedglobal)(IS,PetscBool*);
  PetscErrorCode (*uniquelocal)(IS,PetscBool*);
  PetscErrorCode (*uniqueglobal)(IS,PetscBool*);
  PetscErrorCode (*permlocal)(IS,PetscBool*);
  PetscErrorCode (*permglobal)(IS,PetscBool*);
  PetscErrorCode (*intervallocal)(IS,PetscBool*);
  PetscErrorCode (*intervalglobal)(IS,PetscBool*);
};

typedef enum {IS_INFO_UNKNOWN=0, IS_INFO_FALSE=1, IS_INFO_TRUE=2} ISInfoBool;

struct _p_IS {
  PETSCHEADER(struct _ISOps);
  PetscLayout  map;
  PetscInt     max,min;         /* range of possible values */
  void         *data;
  PetscInt     *total, *nonlocal;   /* local representation of ALL indices across the comm as well as the nonlocal part. */
  PetscInt     local_offset;        /* offset to the local part within the total index set */
  IS           complement;          /* IS wrapping nonlocal indices. */
  PetscBool    info_permanent[2][IS_INFO_MAX]; /* whether local / global properties are permanent */
  ISInfoBool   info[2][IS_INFO_MAX];         /* local / global properties */
};

PETSC_EXTERN PetscErrorCode ISView_Binary(IS, PetscViewer);
PETSC_EXTERN PetscErrorCode ISLoad_Binary(IS, PetscViewer);
PETSC_EXTERN PetscErrorCode ISLoad_Default(IS, PetscViewer);

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
  PetscBool    dealloc_indices; /* should indices be deallocated? */
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
