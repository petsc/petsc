#if !defined(__ASM_H)
#define __ASM_H

#include <petscdm.h>
#include <petsc/private/pcimpl.h>

typedef struct {
  PetscInt   n, n_local, n_local_true;
  PetscInt   overlap;             /* overlap requested by user */
  KSP        *ksp;                /* linear solvers for each block */
  VecScatter restriction;         /* mapping from global to overlapping (process) subdomain*/
  VecScatter *lrestriction;       /* mapping from subregion to overlapping (process) subdomain */
  VecScatter *lprolongation;      /* mapping from non-overlapping subregion to overlapping (process) subdomain; used for restrict additive version of algorithms */
  Vec        lx, ly;              /* work vectors */
  Vec        *x,*y;               /* work vectors */
  IS         lis;                 /* index set that defines each overlapping multiplicative (process) subdomain */
  IS         *is;                 /* index set that defines each overlapping subdomain */
  IS         *is_local;           /* index set that defines each non-overlapping subdomain, may be NULL */
  Mat        *mat,*pmat;          /* mat is not currently used */
  PCASMType  type;                /* use reduced interpolation, restriction or both */
  PetscBool  type_set;            /* if user set this value (so won't change it for symmetric problems) */
  PetscBool  sort_indices;        /* flag to sort subdomain indices */
  PetscBool  dm_subdomains;       /* whether DM is allowed to define subdomains */
  PCCompositeType loctype;        /* the type of composition for local solves */
  MatType    sub_mat_type;        /* the type of Mat used for subdomain solves (can be MATSAME or NULL) */
  /* For multiplicative solve */
  Mat       *lmats;               /* submatrices for overlapping multiplicative (process) subdomain */
} PC_ASM;
#endif
