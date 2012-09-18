#include <petsc-private/kspimpl.h> /*I "petscksp.h" I*/
#include <petscdm.h>         /*I "petscdm.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "PetscContainerDestroy_KSPDM"
static PetscErrorCode PetscContainerDestroy_KSPDM(void *ctx)
{
  PetscErrorCode ierr;
  KSPDM kdm = (KSPDM)ctx;

  PetscFunctionBegin;
  if (kdm->destroy) {ierr = (*kdm->destroy)(kdm);CHKERRQ(ierr);}
  ierr = PetscFree(kdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_KSPDM"
/* Attaches the KSPDM to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMCoarsenHook_KSPDM(DM dm,DM dmc,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMKSPCopyContext(dm,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPGetContext"
/*@C
   DMKSPGetContext - get read-only private KSPDM context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with KSP

   Output Argument:
.  snesdm - private KSPDM context

   Level: developer

   Notes:
   Use DMKSPGetContextWrite() if write access is needed. The DMKSPSetXXX API should be used wherever possible.

.seealso: DMKSPGetContextWrite()
@*/
PetscErrorCode DMKSPGetContext(DM dm,KSPDM *snesdm)
{
  PetscErrorCode ierr;
  PetscContainer container;
  KSPDM         kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)dm,"KSPDM",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)snesdm);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(dm,"Creating new KSPDM\n");CHKERRQ(ierr);
    ierr = PetscContainerCreate(((PetscObject)dm)->comm,&container);CHKERRQ(ierr);
    ierr = PetscNewLog(dm,struct _n_KSPDM,&kdm);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,kdm);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_KSPDM);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"KSPDM",(PetscObject)container);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_KSPDM,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscContainerGetPointer(container,(void**)snesdm);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPGetContextWrite"
/*@C
   DMKSPGetContextWrite - get write access to private KSPDM context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with KSP

   Output Argument:
.  snesdm - private KSPDM context

   Level: developer

.seealso: DMKSPGetContext()
@*/
PetscErrorCode DMKSPGetContextWrite(DM dm,KSPDM *snesdm)
{
  PetscErrorCode ierr;
  KSPDM         kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMKSPGetContext(dm,&kdm);CHKERRQ(ierr);
  if (!kdm->originaldm) kdm->originaldm = dm;
  if (kdm->originaldm != dm) {  /* Copy on write */
    PetscContainer container;
    KSPDM         oldsdm = kdm;
    ierr = PetscInfo(dm,"Copying KSPDM due to write\n");CHKERRQ(ierr);
    ierr = PetscContainerCreate(((PetscObject)dm)->comm,&container);CHKERRQ(ierr);
    ierr = PetscNewLog(dm,struct _n_KSPDM,&kdm);CHKERRQ(ierr);
    ierr = PetscMemcpy(kdm,oldsdm,sizeof(*kdm));CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,kdm);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_KSPDM);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"KSPDM",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  *snesdm = kdm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPCopyContext"
/*@C
   DMKSPCopyContext - copies a DM context to a new DM

   Logically Collective

   Input Arguments:
+  dmsrc - DM to obtain context from
-  dmdest - DM to add context to

   Level: developer

   Note:
   The context is copied by reference. This function does not ensure that a context exists.

.seealso: DMKSPGetContext(), KSPSetDM()
@*/
PetscErrorCode DMKSPCopyContext(DM dmsrc,DM dmdest)
{
  PetscErrorCode ierr;
  PetscContainer container;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)dmsrc,"KSPDM",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscObjectCompose((PetscObject)dmdest,"KSPDM",(PetscObject)container);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dmdest,DMCoarsenHook_KSPDM,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPSetComputeOperators"
/*@C
   DMKSPSetComputeOperators - set KSP matrix evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with KSP
.  func - matrix evaluation function, see KSPSetComputeOperators() for calling sequence
-  ctx - context for matrix evaluation

   Level: advanced

   Note:
   KSPSetComputeOperators() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the matrix.

.seealso: DMKSPSetContext(), DMKSPGetComputeOperators(), KSPSetOperators()
@*/
PetscErrorCode DMKSPSetComputeOperators(DM dm,PetscErrorCode (*func)(KSP,Mat,Mat,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  KSPDM kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMKSPGetContextWrite(dm,&kdm);CHKERRQ(ierr);
  if (func) kdm->computeoperators = func;
  if (ctx)  kdm->operatorsctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPGetComputeOperators"
/*@C
   DMKSPGetComputeOperators - get KSP matrix evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with KSP

   Output Arguments:
+  func - matrix evaluation function, see KSPSetComputeOperators() for calling sequence
-  ctx - context for matrix evaluation

   Level: advanced

.seealso: DMKSPSetContext(), KSPSetComputeOperators(), DMKSPSetComputeOperators()
@*/
PetscErrorCode DMKSPGetComputeOperators(DM dm,PetscErrorCode (**func)(KSP,Mat,Mat,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  KSPDM kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMKSPGetContext(dm,&kdm);CHKERRQ(ierr);
  if (func) *func = kdm->computeoperators;
  if (ctx)  *(void**)ctx = kdm->operatorsctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPSetComputeRHS"
/*@C
   DMKSPSetComputeRHS - set KSP matrix evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with KSP
.  func - right hand side evaluation function, see KSPSetComputeRHS() for calling sequence
-  ctx - context for right hand side evaluation

   Level: advanced

   Note:
   KSPSetComputeRHS() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the matrix.

.seealso: DMKSPSetContext(), DMKSPGetComputeRHS(), KSPSetRHS()
@*/
PetscErrorCode DMKSPSetComputeRHS(DM dm,PetscErrorCode (*func)(KSP,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  KSPDM kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMKSPGetContextWrite(dm,&kdm);CHKERRQ(ierr);
  if (func) kdm->computerhs = func;
  if (ctx)  kdm->rhsctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPGetComputeRHS"
/*@C
   DMKSPGetComputeRHS - get KSP matrix evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with KSP

   Output Arguments:
+  func - right hand side evaluation function, see KSPSetComputeRHS() for calling sequence
-  ctx - context for right hand side evaluation

   Level: advanced

.seealso: DMKSPSetContext(), KSPSetComputeRHS(), DMKSPSetComputeRHS()
@*/
PetscErrorCode DMKSPGetComputeRHS(DM dm,PetscErrorCode (**func)(KSP,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  KSPDM kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMKSPGetContext(dm,&kdm);CHKERRQ(ierr);
  if (func) *func = kdm->computerhs;
  if (ctx)  *(void**)ctx = kdm->rhsctx;
  PetscFunctionReturn(0);
}
