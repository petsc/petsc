/*$Id: dalocal.c,v 1.28 2001/02/02 16:52:57 bsmith Exp balay $*/
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DACreateLocalVector"
/*@C
   DACreateLocalVector - Creates a Seq PETSc vector that
   may be used with the DAXXX routines.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the local vector

   Level: beginner

   Note:
   The output parameter, g, is a regular PETSc vector that should be destroyed
   with a call to VecDestroy() when usage is finished.

.keywords: distributed array, create, local, vector

.seealso: DACreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DAGetLocalVector(), DARestoreLocalVector()
@*/
int DACreateLocalVector(DA da,Vec* g)
{
  int ierr;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecDuplicate(da->local,g);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*g,"DA",(PetscObject)da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetLocalVector"
/*@C
   DAGetLocalVector - Gets a Seq PETSc vector that
   may be used with the DAXXX routines.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the local vector

   Level: beginner

   Note:
   The output parameter, g, is a regular PETSc vector that should be returned with 
   DARestoreLocalVector() DO NOT call VecDestroy() on it.

.keywords: distributed array, create, local, vector

.seealso: DACreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DACreateLocalVector(), DARestoreLocalVector()
@*/
int DAGetLocalVector(DA da,Vec* g)
{
  int ierr,i;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  for (i=0; i<10; i++) {
    if (da->localin[i]) {
      *g             = da->localin[i];
      da->localin[i] = PETSC_NULL;
      goto alldone;
    }
  }
  ierr = VecDuplicate(da->local,g);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*g,"DA",(PetscObject)da);CHKERRQ(ierr);

  alldone:
  for (i=0; i<10; i++) {
    if (!da->localout[i]) {
      da->localout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DARestoreLocalVector"
/*@C
   DARestoreLocalVector - Returns a Seq PETSc vector that
     obtained from DAGetLocalVector(). Do not use with vector obtained via
     DACreateLocalVector().

   Not Collective

   Input Parameter:
+  da - the distributed array
-  g - the local vector

   Level: beginner

.keywords: distributed array, create, local, vector

.seealso: DACreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DACreateLocalVector(), DAGetLocalVector()
@*/
int DARestoreLocalVector(DA da,Vec* g)
{
  int ierr,i,j;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  for (j=0; j<10; j++) {
    if (*g == da->localout[j]) {
      da->localout[j] = PETSC_NULL;
      for (i=0; i<10; i++) {
        if (!da->localin[i]) {
          da->localin[i] = *g;
          goto alldone;
        }
      }
    }
  }
  ierr = VecDestroy(*g);CHKERRQ(ierr);
  alldone:
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetGlobalVector"
/*@C
   DAGetGlobalVector - Gets a MPI PETSc vector that
   may be used with the DAXXX routines.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the global vector

   Level: beginner

   Note:
   The output parameter, g, is a regular PETSc vector that should be returned with 
   DARestoreGlobalVector() DO NOT call VecDestroy() on it.

.keywords: distributed array, create, Global, vector

.seealso: DACreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DACreateLocalVector(), DARestoreLocalVector()
@*/
int DAGetGlobalVector(DA da,Vec* g)
{
  int ierr,i;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  for (i=0; i<10; i++) {
    if (da->globalin[i]) {
      *g             = da->globalin[i];
      da->globalin[i] = PETSC_NULL;
      goto alldone;
    }
  }
  ierr = VecDuplicate(da->global,g);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*g,"DA",(PetscObject)da);CHKERRQ(ierr);

  alldone:
  for (i=0; i<10; i++) {
    if (!da->globalout[i]) {
      da->globalout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DARestoreGlobalVector"
/*@C
   DARestoreGlobalVector - Returns a Seq PETSc vector that
     obtained from DAGetGlobalVector(). Do not use with vector obtained via
     DACreateGlobalVector().

   Not Collective

   Input Parameter:
+  da - the distributed array
-  g - the global vector

   Level: beginner

.keywords: distributed array, create, global, vector

.seealso: DACreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToGlobalBegin(),
          DAGlobalToGlobalEnd(), DAGlobalToGlobal(), DACreateLocalVector(), DAGetGlobalVector()
@*/
int DARestoreGlobalVector(DA da,Vec* g)
{
  int ierr,i,j;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  for (j=0; j<10; j++) {
    if (*g == da->globalout[j]) {
      da->globalout[j] = PETSC_NULL;
      for (i=0; i<10; i++) {
        if (!da->globalin[i]) {
          da->globalin[i] = *g;
          goto alldone;
        }
      }
    }
  }
  ierr = VecDestroy(*g);CHKERRQ(ierr);
  alldone:
  PetscFunctionReturn(0);
}
