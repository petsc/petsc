/*$Id: dapf.c,v 1.5 2000/05/05 22:19:22 balay Exp bsmith $*/
 
#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/


#undef __FUNC__  
#define __FUNC__ "DACreatePF"
/*@
   DACreatePF - Creates an appropriately dimensioned PF mathematical function object
      from a DA.

   Collective on DA

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  pf - the mathematical function object

   Level: advanced


.keywords:  distributed array, grid function

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), DACreateGlobalVector()
@*/
int DACreatePF(DA da,PF *pf)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = PFCreate(da->comm,da->dim,da->w,pf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 

