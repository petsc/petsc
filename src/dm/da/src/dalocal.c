/*$Id: dalocal.c,v 1.29 2001/03/23 23:25:00 balay Exp bsmith $*/
 
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
  for (i=0; i<DA_MAX_WORK_VECTORS; i++) {
    if (da->localin[i]) {
      *g             = da->localin[i];
      da->localin[i] = PETSC_NULL;
      goto alldone;
    }
  }
  ierr = VecDuplicate(da->local,g);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*g,"DA",(PetscObject)da);CHKERRQ(ierr);

  alldone:
  for (i=0; i<DA_MAX_WORK_VECTORS; i++) {
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
  for (j=0; j<DA_MAX_WORK_VECTORS; j++) {
    if (*g == da->localout[j]) {
      da->localout[j] = PETSC_NULL;
      for (i=0; i<DA_MAX_WORK_VECTORS; i++) {
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
  for (i=0; i<DA_MAX_WORK_VECTORS; i++) {
    if (da->globalin[i]) {
      *g             = da->globalin[i];
      da->globalin[i] = PETSC_NULL;
      goto alldone;
    }
  }
  ierr = VecDuplicate(da->global,g);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*g,"DA",(PetscObject)da);CHKERRQ(ierr);

  alldone:
  for (i=0; i<DA_MAX_WORK_VECTORS; i++) {
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
  for (j=0; j<DA_MAX_WORK_VECTORS; j++) {
    if (*g == da->globalout[j]) {
      da->globalout[j] = PETSC_NULL;
      for (i=0; i<DA_MAX_WORK_VECTORS; i++) {
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

/* ------------------------------------------------------------------- */
#if defined(PETSC_HAVE_ADIC)

#include "adic_utils.h"
#undef __FUNCT__
#define __FUNCT__ "DAGetADArray"
/*@
     DAGetADArray - Gets an array of derivative types for a DA
          
    Input Parameter:
+    info - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch

    Output Parameters:
+    ptr - array data structured to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly
-    tdof - total number of degrees of freedom represented in array_start

     Notes: Returns the same type of object as the DAVecGetArray() except its elements are 
           derivative types instead of Scalars

     Level: advanced

.seealso: DARestoreADArray()

@*/
int DAGetADArray(DA da,PetscTruth ghosted,void **iptr,void **array_start,int *tdof)
{
  int  ierr,j,i,deriv_type_size,xs,ys,xm,ym,zs,zm;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->adarrayghostedin[i]) {
        *iptr        = da->adarrayghostedin[i];
        *array_start = da->adstartghostedin[i];
        da->adarrayghostedin[i] = PETSC_NULL;
        da->adstartghostedin[i] = PETSC_NULL;
        goto done;
      }
    }
    xs = da->Xs;
    ys = da->Ys;
    zs = da->Zs;
    xm = da->Xe-da->Xs;
    ym = da->Ye-da->Ys;
    zm = da->Ze-da->Zs;
  } else {
    xs = da->xs;
    ys = da->ys;
    zs = da->zs;
    xm = da->xe-da->xs;
    ym = da->ye-da->ys;
    zm = da->ze-da->zs;
  }
  deriv_type_size = my_AD_GetDerivTypeSize();


  switch (da->dim) {
    case 1: {
      void *ptr;
      *tdof = xm;

      ierr  = PetscMalloc(xm*deriv_type_size,array_start);CHKERRQ(ierr);
      ierr  = PetscMemzero(*array_start,xm*deriv_type_size);CHKERRQ(ierr);

      ptr   = (void*)(*array_start - ys*sizeof(void*));
      *iptr = (void*)ptr; 
      break;}
    case 2: {
      void **ptr;
      *tdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void *)+xm*ym*deriv_type_size,array_start);CHKERRQ(ierr);
      ierr  = PetscMemzero(*array_start,xm*ym*deriv_type_size);CHKERRQ(ierr);

      ptr  = (void**)(*array_start + xm*ym*deriv_type_size - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = *array_start + deriv_type_size*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    case 3: {
      void ***ptr,**bptr;
      *tdof = xm*ym*zm;

      ierr  = PetscMalloc((zm+1)*sizeof(void **)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*deriv_type_size,array_start);CHKERRQ(ierr);
      ierr  = PetscMemzero(*array_start,xm*ym*zm*deriv_type_size);CHKERRQ(ierr);

      ptr  = (void***)(*array_start + xm*ym*zm*deriv_type_size - zs*sizeof(void*));
      bptr = (void**)(*array_start + xm*ym*zm*deriv_type_size + zm*sizeof(void**));
      for(i=zs;i<zs+zm;i++) {
        ptr[i] = bptr + ((i-zs)*ym* - ys)*sizeof(void*);
      }
      for (i=zs; i<zs+zm; i++) {
        for (j=ys; j<ys+ym; j++) {
          ptr[i][j] = *array_start + deriv_type_size*(xm*ym*(i-zs) + xm*(j-ys) - xs);
        }
      }

      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(1,"Dimension %d not supported",da->dim);
  }

  done:
    ;

  PetscFunctionReturn(0);
}

#endif
