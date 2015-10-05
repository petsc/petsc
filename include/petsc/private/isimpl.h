/*
    Index sets for scatter-gather type operations in vectors
and matrices.

*/

#if !defined(_IS_H)
#define _IS_H

#include <petscis.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool ISRegisterAllCalled;
PETSC_EXTERN PetscErrorCode ISRegisterAll(void);

struct _ISOps {
  PetscErrorCode (*getsize)(IS,PetscInt*);
  PetscErrorCode (*getlocalsize)(IS,PetscInt*);
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

struct _p_ISLocalToGlobalMapping{
  PETSCHEADER(int);
  PetscInt  n;                  /* number of local indices */
  PetscInt  bs;                 /* blocksize; there is one index per block */
  PetscInt  *indices;           /* global index of each local index */
  PetscInt  globalstart;        /* first global referenced in indices */
  PetscInt  globalend;          /* last + 1 global referenced in indices */
  PetscInt  *globals;           /* local index for each global index between start and end */
  PetscBool info_cached;        /* reuse GetInfo */
  PetscBool info_free;
  PetscInt  info_nproc;
  PetscInt  *info_procs;
  PetscInt  *info_numprocs;
  PetscInt  **info_indices;
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

/* ----------------------------------------------------------------------------*/
struct _p_PetscSection {
  PETSCHEADER(int);
  PetscInt                      pStart, pEnd; /* The chart: all points are contained in [pStart, pEnd) */
  IS                            perm;         /* A permutation of [0, pEnd-pStart) */
  PetscInt                     *atlasDof;     /* Describes layout of storage, point --> # of values */
  PetscInt                     *atlasOff;     /* Describes layout of storage, point --> offset into storage */
  PetscInt                      maxDof;       /* Maximum dof on any point */
  PetscSection                  bc;           /* Describes constraints, point --> # local dofs which are constrained */
  PetscInt                     *bcIndices;    /* Local indices for constrained dofs */
  PetscBool                     setup;

  PetscInt                      numFields;    /* The number of fields making up the degrees of freedom */
  const char                  **fieldNames;   /* The field names */
  PetscInt                     *numFieldComponents; /* The number of components in each field */
  PetscSection                 *field;        /* A section describing the layout and constraints for each field */

  PetscObject                   clObj;        /* Key for the closure (right now we only have one) */
  PetscSection                  clSection;    /* Section giving the number of points in each closure */
  IS                            clPoints;     /* Points in each closure */
};


#endif
