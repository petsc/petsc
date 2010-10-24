#define PETSCVEC_DLL
/*
   This file contains routines for operations on domain-decomposition-based (DD-based) parallel vectors.
 */
#include "../src/dm/dd/vecdd/ddlayout.h"   
#include "private/isimpl.h"              /*I   "petscis.h"   I*/



#undef __FUNCT__  
#define __FUNCT__ "DDLayoutSetDomainsLocal"
PetscErrorCode DDLayoutSetDomainsLocal(DDLayout dd, PetscInt supported_domain_count, PetscInt *supported_domains, PetscInt *domain_limits, PetscBool covering) {
  PetscErrorCode ierr;
  PetscInt i;
  PetscBool sorted;
  PetscFunctionBegin;

  /* Check args */
  dd->i = supported_domain_count;
  if(dd->i < 0) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative number of locally-supported domains: %d", dd->i); 
  }
  PetscValidPointer(supported_domains, 3);
  PetscValidPointer(domain_limits, 4);
  if(dd->i > 0) {
    /* Make sure the domain list is sorted and,  while at it, determine the highest global domain number. */
    if(supported_domains[0] < 0) {
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative first locally-supported domain index: %d", supported_domains[0]);
    }
    for(i = 1; i < dd->i; ++i) {
      if(supported_domains[i] < supported_domains[i-1]) {
        SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Locally-supported domain indices must be nondegcreasing, "
                 "instead for consecutive domains %d and %d got: %d and %d", i-1,i,supported_domains[i-1],supported_domains[i]);
      }
    }
    dd->dmin = supported_domains[0];
    dd->dmax = supported_domains[dd->i-1];
  }
  else {
    dd->dmin =  0;
    dd->dmax = -1;
  }
  ierr = MPI_Allreduce(&dd->dmin, &dd->Dmin, 1, MPI_INT, MPI_MIN, dd->map->comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&dd->dmax, &dd->Dmax, 1, MPI_INT, MPI_MAX, dd->map->comm); CHKERRQ(ierr);
  if(dd->Dmin < 0) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative global domain lower bound: %d", dd->Dmin);
  }
  /* Make sure domain limit array is sorted, strictly increasing. */
  for(i = 0; i < dd->i; ++i) {
    PetscInt low, high;
    low  = domain_limits[i];
    high = domain_limits[i+1];
    /* Check domain limits */
    /* make sure domain sizes are within range and increasing. */
    if(low < 0 || high < low || high > dd->map->n) {
      SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Domain limits must be positive, not exceeding %d, and nonincreasing; instead at %d and %d got %d and %d", 
               dd->map->n, i, i+1,low,high);                                   
    }
  }
  /* Determine the (local and global) dimension of the space covered by the domains. */
  if(dd->i > 0) {
    dd->n = domain_limits[dd->i]-domain_limits[0];
  }
  else {
    dd->n = 0;
  }
  ierr  = MPI_Allreduce(&dd->n, &dd->N, 1, MPIU_INT, MPI_SUM, dd->map->comm);CHKERRQ(ierr);
  if(covering && (dd->n != dd->map->n || dd->N != dd->map->N)) {
    SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Either local (size %d) or global (size %d) space covered by the domains is less than the unpartitioned space (sizes %d and %d)", dd->n, dd->N, dd->map->n, dd->map->N);
  }
  /** Now create ISs to hold the supported domains and their limits */
  /* Supported domains IS */
  if(dd->locally_supported_domains) {
    ierr = ISDestroy(dd->locally_supported_domains); CHKERRQ(ierr);
  }
  if(dd->i > 0) {
    ierr = ISCreateGeneral(dd->map->comm, dd->i, supported_domains, PETSC_COPY_VALUES, &(dd->locally_supported_domains)); CHKERRQ(ierr);
  }
  else {
    ierr = ISCreateStride(dd->map->comm, 0, 0, 1, &(dd->locally_supported_domains)); CHKERRQ(ierr);
  }
#if defined PETSC_USE_DEBUG
  ierr = ISSorted(dd->locally_supported_domains, &sorted); CHKERRQ(ierr);
  if(!sorted) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected to create a sorted IS of locally-supported domains, but got unsorted");
  }
#endif
  /* Local domain boundaries */
  if(dd->local_domain_limits) {
    ierr = ISDestroy(dd->local_domain_limits); CHKERRQ(ierr);
  }
  if(dd->i > 0) {
    ierr = ISCreateGeneral(dd->map->comm, dd->i+1, domain_limits, PETSC_COPY_VALUES, &(dd->local_domain_limits)); CHKERRQ(ierr);
  }
  else {
    /* Make sure that dd->local_domain_limits has size one bigger than that of dd->locally_supported_domains. */
    ierr = ISCreateStride(dd->map->comm, 1, 0, 1, &(dd->local_domain_limits));                 CHKERRQ(ierr);
  }
#if defined PETSC_USE_DEBUG
  ierr = ISSorted(dd->local_domain_limits, &sorted); CHKERRQ(ierr);
  if(!sorted) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected to create a sorted IS of locally-supported domains, but got unsorted");
  }
#endif
  PetscFunctionReturn(0);
}/* DDLayoutSetDomainsLocal() */


#undef __FUNCT__  
#define __FUNCT__ "DDLayoutSetDomainsLocalIS"
PetscErrorCode DDLayoutSetDomainsLocalIS(DDLayout dd, IS supported_domains, IS domain_limits, PetscBool covering) {
  PetscErrorCode ierr;
  PetscInt i1, lmin, lmax;
  PetscBool sorted;
  PetscFunctionBegin;

  /* Check args */
  ierr = ISGetLocalSize(supported_domains, &dd->i); CHKERRQ(ierr);
  if(dd->i < 0) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative number of locally-supported domains: %d", dd->i); 
  }
  if(dd->i > 0) {
    ierr = ISGetLocalSize(domain_limits, &i1); CHKERRQ(ierr);
    if(dd->i+1 != i1) {
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Domain limits IS size should be %d, got %d instead", dd->i+1, i1); 
    }
    /** Check supported_domains **/
    /* Check sorted. */
    ierr = ISSorted(supported_domains, &sorted); CHKERRQ(ierr);
    if(!sorted) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Supported domains must be a sorted IS");
    }
    /* Check nonnegative. */
    dd->dmin = supported_domains->min;
    dd->dmax = supported_domains->max;
    if(dd->dmin < 0) {
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative first locally-supported domain index: %d", dd->dmin);      
    }
    /** Check domain_limits **/
    /* Check sorted. */
    ierr = ISSorted(domain_limits, &sorted); CHKERRQ(ierr);
    if(!sorted) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Domain limits must be a sorted IS");
    }
    /* Check nonnegative and within local PetscLayout bounds. */
    lmin = domain_limits->min;
    lmax = domain_limits->max;
    if(lmin < 0 || lmax > dd->map->n) {
      SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local domain limits (%d,%d) outside of the (0,%d) bounds", lmin,lmax,dd->map->n);            
    }
  }
  else {
    dd->dmin =  0;
    dd->dmax =  -1;
  }
  ierr = MPI_Allreduce(&dd->dmin, &dd->Dmin, 1, MPI_INT, MPI_MIN, dd->map->comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&dd->dmax, &dd->Dmax, 1, MPI_INT, MPI_MAX, dd->map->comm); CHKERRQ(ierr);
  if(dd->Dmin < 0) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative global domain lower bound: %d", dd->Dmin);
  }
  /**/
  if(dd->locally_supported_domains) {
    ierr = ISDestroy(dd->locally_supported_domains); CHKERRQ(ierr);
  }
  ierr = ISDuplicate(supported_domains, &(dd->locally_supported_domains)); CHKERRQ(ierr);
  if(dd->local_domain_limits) {
    ierr = ISDestroy(dd->local_domain_limits); CHKERRQ(ierr);
  }
  ierr = ISDuplicate(domain_limits,       &(dd->local_domain_limits));     CHKERRQ(ierr);
  /* Compute the dimension of the space covered by the domains; local and global. */
  if(dd->i > 0) {
    dd->n = dd->local_domain_limits->max-dd->local_domain_limits->min;
  }
  else {
    dd->n = 0;
  }
  ierr = MPI_Allreduce(&dd->n, &dd->N, 1, MPIU_INT, MPI_SUM, dd->map->comm);CHKERRQ(ierr);
  if((dd->n > dd->map->n || dd->N > dd->map->N)) {
    SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Either local (size %d) or global (size %d) space covered by the domains is greater than  the unpartitioned space (sizes %d and %d)", dd->n, dd->N, dd->map->n, dd->map->N);
  }
  if(covering && (dd->n < dd->map->n || dd->N < dd->map->N)) {
    SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Either local (size %d) or global (size %d) space covered by the domains is smaller than  the unpartitioned space (sizes %d and %d)", dd->n, dd->N, dd->map->n, dd->map->N);
  }
  PetscFunctionReturn(0);
}/* DDLayoutSetDomainsLocalIS() */


#undef __FUNCT__  
#define __FUNCT__ "DDLayoutDestroy"
PetscErrorCode DDLayoutDestroy(DDLayout dd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(dd && !dd->refcnt--) {
    --((dd->map)->refcnt);
    if(dd->locally_supported_domains) {
      ierr = ISDestroy(dd->locally_supported_domains); CHKERRQ(ierr);
    }
    if(dd->local_domain_limits) {
      ierr = ISDestroy(dd->local_domain_limits); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}/* DDLayoutDestroy() */

#undef __FUNCT__  
#define __FUNCT__ "DDLayoutCreate"
PetscErrorCode DDLayoutCreate(PetscLayout map, DDLayout *dd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNew(struct _n_DDLayout,dd);CHKERRQ(ierr);
  (*dd)->map    = map;
  ++(map->refcnt);
  (*dd)->i    =  -1;
  (*dd)->dmin =  -1;
  (*dd)->dmax =  -1;
  (*dd)->Dmin =  -1;
  (*dd)->Dmax =  -1;
  (*dd)->n    =  -1;
  (*dd)->N    =  -1;
  (*dd)->locally_supported_domains = PETSC_NULL;
  (*dd)->local_domain_limits       = PETSC_NULL;
  PetscFunctionReturn(0);
}/* DDLayoutCreate() */

#undef __FUNCT__  
#define __FUNCT__ "DDLayoutDuplicate"
PetscErrorCode DDLayoutDuplicate(DDLayout ddin, DDLayout *dd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNew(struct _n_DDLayout,dd);CHKERRQ(ierr);
  (*dd)->map    = ddin->map;
  ++(ddin->map->refcnt);
  (*dd)->i    =  ddin->i;
  (*dd)->dmin =  ddin->dmin;
  (*dd)->dmax =  ddin->dmax;
  (*dd)->Dmin =  ddin->Dmin;
  (*dd)->Dmax =  ddin->Dmax;
  (*dd)->n    =  ddin->n;
  (*dd)->N    =  ddin->N;
  if(ddin->locally_supported_domains) {
    ierr = ISDuplicate(ddin->locally_supported_domains, &((*dd)->locally_supported_domains)); CHKERRQ(ierr);
  }
  if(ddin->local_domain_limits) {
    ierr = ISDuplicate(ddin->local_domain_limits,       &((*dd)->local_domain_limits));       CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* DDLayoutDuplicate() */


