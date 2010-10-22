#define PETSCVEC_DLL
/*
   This file contains routines for operations on domain-decomposition-based (DD-based) parallel vectors.
 */
#include "../src/dm/dd/src/vecdd/ddlayout.h"   /*I  "../src/vec/vec/impls/mpi/pvecimpl.h"   I*/



typedef struct {
  Vec_MPI    mpi; /* This has to come first for casting down to Vec_MPI */
  DDLayout   ddlayout;
} Vec_DD;



    


#undef __FUNCT__  
#define __FUNCT__ "VecDDSetDomains"
PetscErrorCode VecDDSetDomains(Vec v, PetscInt dcount, PetscInt **dN) {
  PetscErrorCode ierr;
  Vec_DD *dd = (Vec_DD*) v->data;
  PetscFunctionBegin;
  if(dd->dvec_gotten_count > 0) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Cannot set new domains while domain vecs are gotten");
  }
  ierr = DDLayoutSetDomains(dd->layout, dcount, dN, PETSC_TRUE); CHKERRQ(ierr);
  ierr = VecDD_DestroyDomainVecs(v);                             CHKERRQ(ierr);
  ierr = VecDDSetDomainArrays(v, PETSC_NULL);                    CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* VecDDSetDomains() */



#undef __FUNCT__  
#define __FUNCT__ "VecDDGetDomainArray"
PetscErrorCode VecDDGetDomainArray(Vec v, PetscInt i, PetscScalar **darr)
{
  PetscErrorCode ierr;
  Vec_DD        *dd = (Vec_DD *)v->data;
  PetscFunctionBegin;
  DDLayoutIsSetup(dd->layout);
  *darrs = dd->darrs;
  PetscFunctionReturn(0);
}/* VecDDGetDomainArrays() */


#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_DD_Private"
PetscErrorCode VecDuplicate_DD_Private(Vec w, Vec_DD *dd, Vec *v)
{
  Vec_DD *wdd = (Vec_DD*)w->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDuplicate_MPI(w,v); CHKERRQ(ierr);
  /* Call native constructor. */
  ierr = VecCreate_DD_Private(*v, dd); CHKERRQ(ierr);
  ierr = PetscMemcpy((*v)->ops, w->ops, sizeof(struct _VecOps)); CHKERRQ(ierr);
  /* Copy the layout. */ 
  ierr = DDLayoutDuplicate(wdd->layout, &dd->layout); CHKERRQ(ierr);
  dd->dvec_gotten_count = 0;
  dd->dvec_gotten = 0; 
  dd->darrs = 0;
  dd->dvecs = 0;
  PetscFunctionReturn(0);
}/* VecDuplicate_DD_Private() */

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_DD"
PetscErrorCode VecDuplicate_DD(Vec w, Vec *v)
{
  Vec_DD *dd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Create a native data structure. */
  ierr           = PetscNewLog(w,Vec_DD,&dd);CHKERRQ(ierr);  
  /* Call the native duplicator. */
  ierr = VecDuplicate_DD_Private(w, dd, v); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* VecDuplicate_DD() */

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_DD_Private"
PetscErrorCode VecDestroy_DD_Private(Vec v)
{
  Vec_DD *dd = (Vec_DD*)v->data;
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDD_DestroyDomainVecs(v);  CHKERRQ(ierr);
  if(dd->darrs_allocated) {
    for(i = 0; i < dd->layout->dcount; ++i) {
      ierr = PetscFree(dd->darrs_allocated[i]); CHKERRQ(ierr);
    }
    ierr = PetscFree(dd->darrs_allocated); CHKERRQ(ierr);
  }
  ierr = DDLayoutDestroy(dd->layout); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}/* VecDestroy_DD_Private() */

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_DD"
PetscErrorCode VecDestroy_DD(Vec v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDestroy_DD_Private(v); CHKERRQ(ierr);
  ierr = VecDestroy_MPI(v);        CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* VecDestroy_DD() */


#undef __FUNCT__  
#define __FUNCT__ "VecCreate_DD_Private"
PetscErrorCode VecCreate_DD_Private(Vec v, Vec_DD *dd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* FIX: Vec_MPI* data is PetscNewLog'ged on this v; how can we undo this? */
  dd->mpi        = *((Vec_MPI)(v->data));
  v->data        = (void*)dd;
  /* Now alter some fields filled in by VecMPI's constructor. */
  v->petscnative = PETSC_FALSE;
  v->ops->duplicate = VecDuplicate_DD;
  v->ops->getarray  = VecGetArray_DD;
  v->ops->restorearray = VecRestoreArray_DD;
  v->ops->destroy      = VecDestroy_DD;
  v->ops->create       = 
  /* Initialize DD-specific fields */
  dd->layout = 0;
  dd->dvec_gotten_count = 0;
  dd->dvec_gotten = 0; 
  dd->darrs = 0;
  dd->darrs_allocated = 0;
  dd->dvecs = 0;
}/* VecCreate_DD_Private() */

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_DD"
PetscErrorCode VecCreate_DD(Vec v)
{
  Vec_DD *dd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Use VecMPI's constructor first. */
  ierr           = VecCreate_MPI_Private(v, PETSC_TRUE, 0, 0); CHKERRQ(ierr);
  /* Create a native data structure. */
  ierr           = PetscNewLog(v,Vec_DD,&dd);CHKERRQ(ierr);
  /* Call native constructor. */
  ierr           = VecCreate_DD_Private(v,dd); CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)v,VECDD);CHKERRQ(ierr);
  ierr = PetscPublishAll(v);CHKERRQ(ierr);
}/* VecCreate_DD() */
EXTERN_C_END









