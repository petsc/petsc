#if !defined(__DDLAYOUT_H)
#define __DDLAYOUT_H
/*
   This file contains routines for operations on domain-decomposition-based (DD-based) parallel vectors.
 */

#include "private/isimpl.h"   /*I  "petscis.h"   I*/
#include "private/vecimpl.h"

struct _n_DDLayout {
  PetscLayout map;
  PetscInt    dmin,dmax; /* dmin is the index of the smallest locally-supported domain, dmax is the index of the largest */
  PetscInt    Dmin,Dmax; /* Dmin is the index of the smallest globally-supported domain, Dmax is the index of the largest */
  PetscInt    i;         /* the number of locally-supported domains */
  PetscInt    n,N; /* numbers of degrees of freedom covered by domains, locally and globally, respectively. */
  IS          locally_supported_domains;
  IS          local_domain_limits;
  PetscInt    refcnt;
};
typedef struct _n_DDLayout *DDLayout;

#define DDLayoutIsSetup(ddlayout) \
{\
  if(!ddlayout || ddlayout->i < 0) {\
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DDLayout domains have not been defined yet"); \
  }\
}


#define DDLayoutGetDomainLocal(ddlayout, i, _d, _low, _high)            \
  /*DDLayoutIsSetup(ddlayout);     */                                   \
{                                                                       \
  PetscErrorCode ierr;                                                  \
  PetscBool      stride;                                                \
  if(i < 0) {                                                           \
    /* Set the domain limits at the beginning of the first domain. */   \
    *_low = *_high =  ddlayout->local_domain_limits->min;               \
  }                                                                     \
  else if (i >= ddlayout->i) {                                          \
    /* Set the domain limits at the end of the last domain */           \
    *_low = *_high = ddlayout->local_domain_limits->max;                \
  }                                                                     \
  else {                                                                \
    /* Determine the domain d with local index i. */                                        \
    ierr = ISStride(ddlayout->locally_supported_domains, &stride);  CHKERRQ(ierr);          \
    if(stride) {                                                                            \
      PetscInt first, step;                                                                 \
      ierr = ISStrideGetInfo(ddlayout->locally_supported_domains, &first, &step); CHKERRQ(ierr);  \
      *_d = first + i*step;                                                                 \
    }                                                                                       \
    else {                                                                                  \
      const PetscInt *idx;                                                                        \
      ierr = ISGetIndices(ddlayout->locally_supported_domains, &idx); CHKERRQ(ierr);        \
      *_d = idx[i];                                                                         \
    }                                                                                       \
    /* Now get the domain limits */                                                  \
    ierr = ISStride(ddlayout->local_domain_limits, &stride); CHKERRQ(ierr);          \
    if(stride) {                                                                     \
      PetscInt first, step;                                                          \
      ierr = ISStrideGetInfo(ddlayout->local_domain_limits, &first, &step); CHKERRQ(ierr); \
      *_low = first + i*step;                                                        \
      *_high = *_low + step;                                                         \
    }                                                                                \
    else { /* non-stride */                                                          \
      const PetscInt *idx;                                                                 \
      ierr = ISGetIndices(ddlayout->local_domain_limits, &idx); CHKERRQ(ierr);      \
      *_low = idx[i]; *_high = idx[i+1];                                             \
    }                                                                                \
  }                                                                                  \
} 

EXTERN PetscErrorCode DDLayoutSetDomainsLocal(DDLayout dd, PetscInt supported_domain_count, PetscInt *supported_domains, PetscInt *domain_limits, PetscBool covering);
EXTERN PetscErrorCode DDLayoutSetDomainsLocalIS(DDLayout dd, IS supported_domains, IS domain_limits, PetscBool covering);
EXTERN PetscErrorCode DDLayoutDestroy(DDLayout dd);
EXTERN PetscErrorCode DDLayoutCreate(PetscLayout map, DDLayout *dd);
EXTERN PetscErrorCode DDLayoutDuplicate(DDLayout ddin, DDLayout *dd);
#endif
