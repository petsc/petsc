
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dadist.c,v 1.20 1999/03/07 17:30:00 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/


#undef __FUNC__  
#define __FUNC__ "DAGetGlobalToGlobal1_Private"
int DAGetGlobalToGlobal1_Private(DA da,int **gtog1)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *gtog1 = da->gtog1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DACreateGlobalVector"
/*@C
   DACreateGlobalVector - Creates a parallel PETSc vector that
   may be used with the DAXXX routines.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the distributed global vector

   Level: beginner

   Note:
   The output parameter, g, is a regular PETSc vector that should be destroyed
   with a call to VecDestroy() when usage is finished.

.keywords: distributed array, create, global, distributed, vector

.seealso: DACreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DACreateNaturalVector()
@*/
int DACreateGlobalVector(DA da,Vec* g)
{
  int ierr;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (da->globalused) {
    ierr = VecDuplicate(da->global,g);CHKERRQ(ierr);
  } else {
    /* 
     compose the DA into the vectors so they have access to the 
     distribution information. 
    */
    ierr = PetscObjectCompose((PetscObject)da->global,"DA",(PetscObject)da);CHKERRQ(ierr);
    da->globalused = PETSC_TRUE;
    *g   = da->global;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DACreateNaturalVector"
/*@C
   DACreateNaturalVector - Creates a parallel PETSc vector that
       will hold vector values in the natural numbering, rather than in 
       the PETSc parallel numbering associated with the DA.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the distributed global vector

   Level: developer

   Note:
   The output parameter, g, is a regular PETSc vector that should be destroyed
   with a call to VecDestroy() when usage is finished.

.keywords: distributed array, create, global, distributed, vector

.seealso: DACreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
int DACreateNaturalVector(DA da,Vec* g)
{
  int cnt,m,ierr;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (da->natural) {
    ierr = PetscObjectGetReference((PetscObject)da->natural,&cnt);CHKERRQ(ierr);
    if (cnt == 1) { /* object is not currently used by anyone */
      ierr = PetscObjectReference((PetscObject)da->natural);CHKERRQ(ierr);
      *g   = da->natural;
    } else {
      ierr = VecDuplicate(da->natural,g);CHKERRQ(ierr);
    }
  } else { /* create the first version of this guy */
    ierr = VecGetLocalSize(da->global,&m);CHKERRQ(ierr);
    ierr = VecCreateMPI(da->comm,m,PETSC_DETERMINE,g);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*g);CHKERRQ(ierr);
    da->natural = *g;
  }
  PetscFunctionReturn(0);
}



