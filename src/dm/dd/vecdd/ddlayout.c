#define PETSCVEC_DLL
/*
   This file contains routines for operations on domain-decomposition-based (DD-based) parallel vectors.
 */
#include "../src/vec/vec/impls/dd/ddlayout.h"   




#undef __FUNCT__  
#define __FUNCT__ "DDLayoutSetDomains"
PetscErrorCode PETSCVEC_DLLEXPORT DDLayoutSetDomains(DDLayout dd, PetscInt supported_domain_count, PetscInt *supported_domains, PetscInt *domain_limits, PetscTruth covering) {
  PetscErrorCode ierr;
  PetscInt maxd, i;
  PetscTurth sorted;
  PetscFunctionBegin;

  /* Check args */
  if(supported_domain_count < 0) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative number of locally-supported domains is illegal: %d", supported_domain_count); 
  }
  PetscValidPointer(supported_domains, 3);
  PetscValidPointer(domain_limits, 4);

  /* Make sure the domain list is sorted and,  while at it, determine the highest global domain number. */
  dd->d = supported_domain_count;
  if(supported_domains[0] < 0) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative first supported domain number: %d", supported_domains[0]);
  }
  for(i = 1; i < dd->d; ++i) {
    if(supported_domains[i] <= supported_domains[i-1]) {
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Locally-supported domain numbers must be increasing, instead got for domains %d and %d: %d and %d", 
               i-1,i,supported_domains[i-1],supported_domains[i]);
    }
  }
  maxd = supported_domains[dd->d-1];
  ierr = MPI_Allreduce(&maxd, &dd->D, 1, MPI_INT, MPI_MAX, dd->map->comm); CHKERRQ(ierr);

  /* 
     Make sure domain limit array is sorted, strictly increasing and, while at it, 
     determine the (local and global) dimension of the space covered by the domains. 
  */
  dd->n = 0;
  for(i = 0; i < dd->d; ++i) {
    PetscInt low, high;
    low  = domain_limits[i];
    high = domain_limits[i+1];
    /* Check domain limits */
    /* make sure domain sizes are within range and increasing. */
    if(low < 0 || high < low || high > dd->map->n) {
      SETERRQ4(PETSC_ERR_USER, "Domain limits must be positive, not exceeding %d, and strictly increasing; instead got at %d and %d", 
               dd->map->n, i, i+1,,low,high);                                   
    }
    dd->n += high-low;
  }
  ierr = MPI_Allreduce(&dd->n, &dd->N, 1, MPIU_INT, MPI_ADD, dd->map->comm);CHKERRQ(ierr);
  if(covering && (dd->n != dd->map->n || dd->N != dd->map->N)) {
    SETERRQ4(PETSC_ERR_ARG_WRONG, "Either local (size %d) or global (size %d) space covered by the domains is less than the unpartitioned space (sizes %d and %d)", dd->n, dd->N, dd->map->n, dd->map->N);
  }
  /** Now create ISs to hold the supported domains and their limits */
  /* Supported domains IS */
  if(dd->locally_supported_domains) {
    ierr = ISDestroy(dd->locally_supported_domains); CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(dd->map->comm, dd->d, supported_domains, &(dd->locally_supported_domains)); CHKERRQ(ierr);
  ierr = ISSorted(dd->locally_supported_domains, &sorted); CHKERRQ(ierr);
  if(!sorted) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected to create a sorted IS of locally-supported domains, but got unsorted");
  }
  /* Local domain boundaries */
  if(dd->local_domain_limits) {
    ierr = ISDestroy(dd->local_domain_limits); CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(dd->map->comm, dd->d+1, domain_limits, &(dd->local_domain_limits)); CHKERRQ(ierr);
  ierr = ISSorted(dd->local_domain_limits, &sorted); CHKERRQ(ierr);
  if(!sorted) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected to create a sorted IS of locally-supported domains, but got unsorted");
  }
  PetscFunctionReturn(0);
}/* DDLayoutSetDomains() */


#undef __FUNCT__  
#define __FUNCT__ "DDLayoutSetDomainsIS"
PetscErrorCode PETSCVEC_DLLEXPORT DDLayoutSetDomainsIS(DDLayout dd, IS locally_supported_domains, IS local_domain_limits, PetscTruth covering) {
  PetscErrorCode ierr;
  PetscInt d, d1, maxd;
  PetscTruth sorted, stride;
  PetscFunctionBegin;

  /* Check args */
  ierr = ISGetLocalSize(supported_domains, &d); CHKERRQ(ierr);
  if(d < 0) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Negative number of locally-supported domains is illegal: %d", d); 
  }
  ierr = ISGetLocalSize(domain_limits, &d1); CHKERRQ(ierr);
  if(d+1 != d1) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Domain limits IS size should be %d, got %d instead: %d", d+1, d1); 
  }
  ierr = ISSorted(supported_domains, &sorted); CHKERRQ(ierr);
  if(!sorted) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Supported domains must be a sorted IS");
  }
  ierr = ISSorted(domain_limits, &sorted); CHKERRQ(ierr);
  if(!sorted) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Domain limits must be a sorted IS");
  }
  /**/
  ierr = ISStride(supported_domains, stride); CHKERRQ(ierr);
  if(stride) {
    PetscInt first, step;
    ierr = ISStrideGetInfo(supported_domains, &first, &step); CHKERRQ(ierr);
    maxd = first + step*(d-1);
  }
  else {
    PetscInt *idx;
    ierr = ISSort(locally_supported_domains);                  CHKERRQ(ierr);
    ierr = ISGetLocalIndices(locally_supported_domains, &idx); CHKERRQ(ierr);
    maxd = idx[d-1];
  }
  dd->d = d;
  ierr = MPI_Allreduce(&maxd, &dd->D, 1, MPI_INT, MPI_MAX, dd->map->comm); CHKERRQ(ierr);
  if(dd->locally_supported_domains) {
    ierr = ISDestroy(dd->locally_supported_domains); CHKERRQ(ierr);
  }
  dd->locally_supported_domains = locally_supported_domains;
  if(dd->local_domain_limits) {
    ierr = ISDestroy(dd->local_domain_limits); CHKERRQ(ierr);
  }
  dd->local_domain_limits = local_domain_limits;
  /* Compute the dimension of the space covered by the domains; local and global */
  ierr = ISStride(dd->local_domain_limits, &stride); CHKERRQ(ierr);
  if(stride) {
    PetscInt first, step;
    ierr = ISStrideGetInfo(dd->local_domain_limits, &first, &step); CHKERRQ(ierr);
    dd->n = step*dd->d;
  }
  else {
    PetscInt *idx;
    ierr = ISGetLocalIndices(dd->local_domain_limits, &idx); CHKERRQ(ierr);
    dd->n = idx[dd->d+1]-idx[dd->d];
  }
  ierr = MPI_Allreduce(&dd->n, &dd->N, 1, MPIU_INT, MPI_ADD, dd->map->comm);CHKERRQ(ierr);
  if((dd->n > dd->map->n || dd->N > dd->map->N)) {
    SETERRQ4(PETSC_ERR_ARG_WRONG, "Either local (size %d) or global (size %d) space covered by the domains is greater than  the unpartitioned space (sizes %d and %d)", dd->n, dd->N, dd->map->n, dd->map->N);
  }
  if((dd->n < dd->map->n || dd->N < dd->map->N) && covering) {
    SETERRQ4(PETSC_ERR_ARG_WRONG, "Either local (size %d) or global (size %d) space covered by the domains is smaller than  the unpartitioned space (sizes %d and %d)", dd->n, dd->N, dd->map->n, dd->map->N);
  }
  PetscFunctionReturn(0);
}/* DDLayoutSetDomainsIS() */


#undef __FUNCT__  
#define __FUNCT__ "DDLayoutDestroy"
PetscErrorCode PETSCVEC_DLLEXPORT DDLayoutDestroy(DDLayout dd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(dd && !dd->refcnt--) {
    --((*(dd->map))->refcnt);
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
PetscErrorCode PETSCVEC_DLLEXPORT DDLayoutCreate(PetscLayout map, DDLayout *dd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNew(struct _n_DDLayout,dd);CHKERRQ(ierr);
  (*dd)->map    = map;
  ++(map->refcnt);
  (*dd)->d =  -1;
  (*dd)->D =  0;
  (*dd)->n =  0;
  (*dd)->N =  0;
  (*dd)->locally_supported_domains = PETSC_NULL;
  (*dd)->local_domain_limits       = PETSC_NULL;
  PetscFunctionReturn(0);
}/* DDLayoutCreate() */

#undef __FUNCT__  
#define __FUNCT__ "DDLayoutDuplicate"
PetscErrorCode PETSCVEC_DLLEXPORT DDLayoutDuplicate(DDLayout ddin, DDLayout *dd)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNew(struct _n_DDLayout,dd);CHKERRQ(ierr);
  (*dd)->map    = ddin->map;
  ++(ddin->map->refcnt);
  (*dd)->d =  ddin->d;
  (*dd)->D =  ddin->D;
  (*dd)->n =  ddin->n;
  (*dd)->N =  ddin->N;
  if(ddin->locally_supported_domains) {
    ierr = ISDuplicate(ddin->locally_supported_domains, &((*dd)->locally_supported_domains)); CHKERRQ(ierr);
  }
  if(ddin->local_domain_limits) {
    ierr = ISDuplicate(ddin->local_domain_limits,       &((*dd)->local_domain_limits));       CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* DDLayoutDuplicate() */


