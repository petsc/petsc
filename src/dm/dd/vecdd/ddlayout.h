#if !defined(__DDLAYOUT_H)
#define __DDLAYOUT_H
/*
   This file contains routines for operations on domain-decomposition-based (DD-based) parallel vectors.
 */
#include "../src/vec/vec/impls/mpi/pvecimpl.h"   /*I  "petscvec.h"   I*/

typedef struct _n_DDLayout *DDLayout;
struct _n_DDLayout {
  PetscLayout map;
  PetscInt    d,D; /* locally-/globally-supported domain counts */
  PetscInt    n,N; /* numbers of degrees of freedom covered by domains, locally and globally, respectively. */
  IS          locally_supported_domains;
  IS          local_domain_limits;
  PetscInt    refcnt;
};

#define DDLayoutIsSetup(ddlayout) \
{\
  PetscErrorCode ierr;\
  if(!ddlayout || ddlayout->d < 0) {\
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "DDLayout domains have not been defined yet");\
  }\
}


#define DDLayoutGetDomain(ddlayout, i, _d, _low, _high)                 \
{                                                                       \
  PetscErrorCode ierr;                                                  \
  PetscTruth     stride;                                                \
  PetscInt       size,d,step;                                           \
  IS             domains;                                               \
  DDLayoutIsSetup(ddlayout);                                            \
  if(i >= ddlayout->d) {                                                \
    *_low = *_high = 0;                                                 \
  }                                                                     \
  /* Determine the domain d with local index i. */                      \
  ierr = ISStride(ddlayout->locally_supported_domains, &flag);  CHKERRQ(ierr); \
  if(flag) { /* stride */                                                                 \ 
    PetscInt first, step;                                                                 \
    ierr = ISGetInfo(ddlyaout->locally_supported_domains, &first, &step); CHKERRQ(ierr);  \
    d = first + i*step;                                                                   \
  }                                                                                       \
  else {                                                                          \
    PetscInt *idx;                                                                \
    ierr = ISGetIndices(ddlyaout->locally_supported_domains, &idx); CHKERRQ(ierr);\
    d = idx[i];                                                                   \
  }                                                                               \
  *_d = d;                                                                        \
  /* Now get the domain limits */                   \
  domains = ddlayout->local_domain_limits;          \
  ierr = ISStride(domains, &stride); CHKERRQ(ierr); \
  if(stride) {                                                \
    PetscInt first, step;                                     \
    ierr = ISGetInfo(domains, &first, &step); CHKERRQ(ierr);  \
    *_low = first + i*step;                                   \
    *_high = *_low + step;                                    \
  }                                                           \
  else { /* non-stride */                                     \
    PetscInt *idx;                                            \
    ierr = ISGetIndices(domains, &idx); CHKERRQ(ierr);        \ 
    *_low = idx[i]; *_high = idx[i+1];                        \
  }                                                           \
} 

#endif
