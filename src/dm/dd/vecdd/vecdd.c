#define PETSCVEC_DLL
/*
   This file contains routines for operations on domain-decomposition-based (DD-based) parallel vectors.
 */
#include "../src/dm/dd/vecdd/ddlayout.h"         
#include "../src/vec/vec/impls/mpi/pvecimpl.h"   /*I  "petscvec.h"   I*/

EXTERN PetscErrorCode VecDuplicate_MPI(Vec in, Vec *out);

typedef struct {
  Vec_MPI    mpi; /* This has to come first for casting down to Vec_MPI */
  DDLayout   layout;
} Vec_DD;


#undef __FUNCT__  
#define __FUNCT__ "VecDDSetDomainsLocal"
PetscErrorCode VecDDSetDomainsLocal(Vec v, PetscInt domain_count, PetscInt supported_domains[], PetscInt domain_limits[], PetscBool covering){
  PetscErrorCode ierr;
  Vec_DD *dd = (Vec_DD*) v->data;
  PetscFunctionBegin;
  ierr = DDLayoutSetDomainsLocal(dd->layout, domain_count, supported_domains, domain_limits, covering); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* VecDDSetDomainsLocal() */

#undef __FUNCT__  
#define __FUNCT__ "VecDDSetDomainsLocalIS"
PetscErrorCode VecDDSetDomainsLocalIS(Vec v, PetscInt domain_count, IS supported_domains, IS domain_limits, PetscBool covering){
  PetscErrorCode ierr;
  Vec_DD *dd = (Vec_DD*) v->data;
  PetscFunctionBegin;
  ierr = DDLayoutSetDomainsLocalIS(dd->layout, supported_domains, domain_limits, covering); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* VecDDSetDomainsLocalIS() */



#undef __FUNCT__  
#define __FUNCT__ "VecDDGetDomainInfoLocal"
PetscErrorCode VecDDGetDomainInfoLocal(Vec v, PetscInt i, PetscInt *_d, PetscInt *_size)
{
  PetscInt       d, low, high;
  Vec_DD        *dd = (Vec_DD *)v->data;
  PetscFunctionBegin;
  DDLayoutGetDomainLocal((dd->layout), i, &d, &low, &high);
  if(!_d) {
    *_d = d;
  }
  if(!_size) {
    *_size = high-low;
  }
  PetscFunctionReturn(0);
}/* VecDDGetDomainInfoLocal() */


#undef __FUNCT__  
#define __FUNCT__ "VecDDGetDomainArrayLocal"
PetscErrorCode VecDDGetDomainArrayLocal(Vec v, PetscInt i, PetscScalar **darr)
{
  PetscInt high, low, d;
  Vec_DD        *dd = (Vec_DD *)v->data;
  PetscFunctionBegin;
  DDLayoutGetDomainLocal(dd->layout, i, &d, &low, &high);
  *darr = (dd->mpi).array + low;
  PetscFunctionReturn(0);
}/* VecDDGetDomainArrayLocal() */

PetscErrorCode VecCreate_DD(Vec v);
PetscErrorCode VecCreate_DD_Private(Vec v, Vec_DD *dd);

/* See the comment in front of VecCreate_DD_Private below. */
#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_DD_Private"
PetscErrorCode VecDuplicate_DD_Private(Vec w, Vec_DD *dd, Vec *v)
{
  Vec_DD *wdd = (Vec_DD*)w->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Call the native constructor. */
  ierr = VecCreate_DD_Private(*v, dd); CHKERRQ(ierr);
  /* Copy the layout. */ 
  ierr = DDLayoutDuplicate(wdd->layout, &dd->layout); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* VecDuplicate_DD_Private() */

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_DD"
PetscErrorCode VecDuplicate_DD(Vec w, Vec *v)
{
  Vec_DD *dd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Call the VecMPI duplicator first. */
  ierr = VecDuplicate_MPI(w,v); CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;
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


/* 
   This function takes a VecMPI v and an empty Vec_DD implementation struct dd.
   The Vec_MPI impl of v, mpi, is then replaced with dd, after copying mpi into 
   dd's Vec_MPI header. The rest of the dd is then constructed and the Vec  vtable, 
   filled in by the VecMPI constructor, is fixed up to use the VecDD dispatch.
     Why pass dd in as an argument?
   This is to allow Vec implementations that extend VecDD to pass in their own
   impl structs, which include Vec_DD as a header and, hence, can be cast to Vec_DD.
   That way VecDD constructs its portion directly in the extending struct, instead 
   of having to copy the Vec_DD data into the extending struct later.
   Had VecCreate_MPI_Private been written that way too, we would be able to avoid
   the copy of mpi into dd.
     Why not have VecCreate_DD_Private call the VecMPI constructor?
   This is to allow different callers to construct the VecMPI vec using different 
   constructors.  For example, VecDuplicate_DD uses VecDuplicate_MPI.
*/
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_DD_Private"
PetscErrorCode VecCreate_DD_Private(Vec v, Vec_DD *dd)
{

  PetscFunctionBegin;
  /* FIX: Vec_MPI* data is PetscNewLog'ged on this v; how can we undo this? */
  dd->mpi        = *((Vec_MPI*)(v->data));
  v->data        = (void*)dd;
  /* Now alter some fields filled in by VecMPI's constructor. */
  v->ops->duplicate    = VecDuplicate_DD;
  v->ops->destroy      = VecDestroy_DD;
  v->ops->create       = VecCreate_DD;
  /* Initialize DD-specific fields */
  dd->layout = 0;
  PetscFunctionReturn(0);
}/* VecCreate_DD_Private() */

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_DD"
PetscErrorCode VecCreate_DD(Vec v)
{
  Vec_DD *dd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Call VecMPI's constructor first. */
  ierr           = VecCreate_MPI_Private(v, PETSC_TRUE, 0, 0); CHKERRQ(ierr);
  /* Create an empty native data structure. */
  ierr           = PetscNewLog(v,Vec_DD,&dd);CHKERRQ(ierr);
  /* Call the native constructor. */
  ierr           = VecCreate_DD_Private(v,dd); CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECDD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* VecCreate_DD() */
EXTERN_C_END









