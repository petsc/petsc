#if !defined(PETSCSECTIONIMPL_H)
#define PETSCSECTIONIMPL_H

#include <petscsection.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/hashmap.h>

typedef struct PetscSectionClosurePermKey {
  PetscInt    depth, size;
} PetscSectionClosurePermKey;

typedef struct {
  PetscInt *perm, *invPerm;
} PetscSectionClosurePermVal;

static inline PetscHash_t PetscSectionClosurePermHash(PetscSectionClosurePermKey k)
{
  return PetscHashCombine(PetscHashInt(k.depth), PetscHashInt(k.size));
}

static inline int PetscSectionClosurePermEqual(PetscSectionClosurePermKey k1, PetscSectionClosurePermKey k2)
{
  return k1.depth == k2.depth && k1.size == k2.size;
}

static PetscSectionClosurePermVal PetscSectionClosurePermVal_Empty = {NULL, NULL};
PETSC_HASH_MAP(ClPerm, PetscSectionClosurePermKey, PetscSectionClosurePermVal, PetscSectionClosurePermHash, PetscSectionClosurePermEqual, PetscSectionClosurePermVal_Empty)

struct _p_PetscSection {
  PETSCHEADER(int);
  PetscInt                      pStart, pEnd; /* The chart: all points are contained in [pStart, pEnd) */
  IS                            perm;         /* A permutation of [0, pEnd-pStart) */
  PetscBool                     pointMajor;   /* True if the offsets are point major, otherwise they are fieldMajor */
  PetscBool                     includesConstraints; /* True if constrained dofs are included when computing offsets */
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
  char                        ***compNames;   /* The component names */

  PetscObject                   clObj;        /* Key for the closure (right now we only have one) */
  PetscClPerm                   clHash;       /* Hash of (depth, size) to perm and invPerm */
  PetscSection                  clSection;    /* Section giving the number of points in each closure */
  IS                            clPoints;     /* Points in each closure */
  PetscSectionSym               sym;          /* Symmetries of the data */
};

struct _PetscSectionSymOps {
  PetscErrorCode (*getpoints)(PetscSectionSym,PetscSection,PetscInt,const PetscInt *,const PetscInt **,const PetscScalar **);
  PetscErrorCode (*distribute)(PetscSectionSym,PetscSF,PetscSectionSym*);
  PetscErrorCode (*copy)(PetscSectionSym,PetscSectionSym);
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

PETSC_EXTERN PetscErrorCode PetscSectionSetClosurePermutation_Internal(PetscSection, PetscObject, PetscInt, PetscInt, PetscCopyMode, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSectionGetClosurePermutation_Internal(PetscSection, PetscObject, PetscInt, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode PetscSectionGetClosureInversePermutation_Internal(PetscSection, PetscObject, PetscInt, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode ISIntersect_Caching_Internal(IS, IS, IS *);
#if defined(PETSC_HAVE_HDF5)
PETSC_INTERN PetscErrorCode PetscSectionView_HDF5_Internal(PetscSection, PetscViewer);
PETSC_INTERN PetscErrorCode PetscSectionLoad_HDF5_Internal(PetscSection, PetscViewer);
#endif

static inline PetscErrorCode PetscSectionCheckConstraints_Private(PetscSection s)
{
  PetscFunctionBegin;
  if (!s->bc) {
    PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s->bc));
    PetscCall(PetscSectionSetChart(s->bc, s->pStart, s->pEnd));
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_CLANG_STATIC_ANALYZER)
#  define PetscSectionCheckValidField(x,y)
#  define PetscSectionCheckValidFieldComponent(x,y)
#else
#  define PetscSectionCheckValid_(description,item,nitems) do {         \
    PetscCheck(((item) >= 0) && ((item) < (nitems)),PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,"Invalid " description " %" PetscInt_FMT "; not in [0, %" PetscInt_FMT ")", (item), (nitems)); \
  } while (0)

#  define PetscSectionCheckValidFieldComponent(comp,nfieldcomp) PetscSectionCheckValid_("section field component",comp,nfieldcomp)

#  define PetscSectionCheckValidField(field,nfields)            PetscSectionCheckValid_("field number",field,nfields)

#endif /* PETSC_CLANG_STATIC_ANALYZER */

#endif /* PETSCSECTIONIMPL_H */
