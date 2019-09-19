#if !defined(PETSCSECTIONIMPL_H)
#define PETSCSECTIONIMPL_H

#include <petscsection.h>
#include <petsc/private/petscimpl.h>

struct _p_PetscSection {
  PETSCHEADER(int);
  PetscInt                      pStart, pEnd; /* The chart: all points are contained in [pStart, pEnd) */
  IS                            perm;         /* A permutation of [0, pEnd-pStart) */
  PetscBool                     pointMajor;   /* True if the offsets are point major, otherwise they are fieldMajor */
  PetscInt                     *atlasDof;     /* Describes layout of storage, point --> # of values */
  PetscInt                     *atlasOff;     /* Describes layout of storage, point --> offset into storage */
  PetscInt                      maxDof;       /* Maximum dof on any point */
  PetscSection                  bc;           /* Describes constraints, point --> # local dofs which are constrained */
  PetscInt                     *bcIndices;    /* Local indices for constrained dofs */
  PetscBool                     setup;

  PetscInt                      numFields;    /* The number of fields making up the degrees of freedom */
  char                        **fieldNames;   /* The field names */
  PetscInt                     *numFieldComponents; /* The number of components in each field */
  PetscSection                 *field;        /* A section describing the layout and constraints for each field */
  PetscBool                     useFieldOff;  /* Use the field offsets directly for the global section, rather than the point offset */

  PetscObject                   clObj;        /* Key for the closure (right now we only have one) */
  PetscSection                  clSection;    /* Section giving the number of points in each closure */
  IS                            clPoints;     /* Points in each closure */
  PetscInt                      clSize;       /* The size of a dof closure of a cell, when it is uniform */
  PetscInt                     *clPerm;       /* A permutation of the cell dof closure, of size clSize */
  PetscInt                     *clInvPerm;    /* The inverse of clPerm */
  PetscSectionSym               sym;          /* Symmetries of the data */
};

struct _PetscSectionSymOps {
  PetscErrorCode (*getpoints)(PetscSectionSym,PetscSection,PetscInt,const PetscInt *,const PetscInt **,const PetscScalar **);
  PetscErrorCode (*destroy)(PetscSectionSym);
  PetscErrorCode (*view)(PetscSectionSym,PetscViewer);
};

typedef struct _n_SymWorkLink *SymWorkLink;

struct _n_SymWorkLink
{
  SymWorkLink         next;
  const PetscInt    **perms;
  const PetscScalar **rots;
  PetscInt           numPoints;
};

struct _p_PetscSectionSym {
  PETSCHEADER(struct _PetscSectionSymOps);
  void *data;
  SymWorkLink workin;
  SymWorkLink workout;
};

PETSC_EXTERN PetscErrorCode PetscSectionSetClosurePermutation_Internal(PetscSection, PetscObject, PetscInt, PetscCopyMode, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSectionGetClosurePermutation_Internal(PetscSection, PetscObject, PetscInt *, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode PetscSectionGetClosureInversePermutation_Internal(PetscSection, PetscObject, PetscInt *, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode ISIntersect_Caching_Internal(IS, IS, IS *);

#endif
