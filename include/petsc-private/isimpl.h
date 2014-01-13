/*
    Index sets for scatter-gather type operations in vectors
and matrices.

*/

#if !defined(_IS_H)
#define _IS_H

#include <petscis.h>
#include <petsc-private/petscimpl.h>

struct _ISOps {
  PetscErrorCode (*getsize)(IS,PetscInt*);
  PetscErrorCode (*getlocalsize)(IS,PetscInt*);
  PetscErrorCode (*getindices)(IS,const PetscInt*[]);
  PetscErrorCode (*restoreindices)(IS,const PetscInt*[]);
  PetscErrorCode (*invertpermutation)(IS,PetscInt,IS*);
  PetscErrorCode (*sort)(IS);
  PetscErrorCode (*sorted)(IS,PetscBool*);
  PetscErrorCode (*duplicate)(IS,IS*);
  PetscErrorCode (*destroy)(IS);
  PetscErrorCode (*view)(IS,PetscViewer);
  PetscErrorCode (*identity)(IS,PetscBool*);
  PetscErrorCode (*copy)(IS,IS);
  PetscErrorCode (*togeneral)(IS);
  PetscErrorCode (*oncomm)(IS,MPI_Comm,PetscCopyMode,IS*);
  PetscErrorCode (*setblocksize)(IS,PetscInt);
  PetscErrorCode (*contiguous)(IS,PetscInt,PetscInt,PetscInt*,PetscBool*);
};

struct _p_IS {
  PETSCHEADER(struct _ISOps);
  PetscBool    isperm;          /* if is a permutation */
  PetscInt     max,min;         /* range of possible values */
  PetscInt     bs;              /* block size */
  void         *data;
  PetscBool    isidentity;
  PetscInt     *total, *nonlocal;   /* local representation of ALL indices across the comm as well as the nonlocal part. */
  PetscInt     local_offset;        /* offset to the local part within the total index set */
  IS           complement;          /* IS wrapping nonlocal indices. */
};

struct _p_ISLocalToGlobalMapping{
  PETSCHEADER(int);
  PetscInt n;                  /* number of local indices */
  PetscInt *indices;           /* global index of each local index */
  PetscInt globalstart;        /* first global referenced in indices */
  PetscInt globalend;          /* last + 1 global referenced in indices */
  PetscInt *globals;           /* local index for each global index between start and end */
};

/* ----------------------------------------------------------------------------*/
typedef struct _n_PetscUniformSection *PetscUniformSection;
struct _n_PetscUniformSection {
  MPI_Comm comm;
  PetscInt pStart, pEnd; /* The chart: all points are contained in [pStart, pEnd) */
  PetscInt numDof;       /* Describes layout of storage, point --> (constant # of values, (p - pStart)*constant # of values) */
};

struct _p_PetscSection {
  PETSCHEADER(int);
  struct _n_PetscUniformSection atlasLayout;  /* Layout for the atlas */
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

  PetscObject                   clObj;        /* Key forthe closure (right now we only have one) */
  PetscSection                  clSection;    /* Section for the closure index */
  IS                            clIndices;    /* Indices for the closure index */
};


#endif
