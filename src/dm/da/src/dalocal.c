#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

/*
   This allows the DA vectors to properly tell Matlab their dimensions
*/
#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include "engine.h"   /* Matlab include file */
#include "mex.h"      /* Matlab include file */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecMatlabEnginePut_DA2d"
PetscErrorCode PETSCDM_DLLEXPORT VecMatlabEnginePut_DA2d(PetscObject obj,void *mengine)
{
  PetscErrorCode ierr;
  PetscInt       n,m;
  Vec            vec = (Vec)obj;
  PetscScalar    *array;
  mxArray        *mat;
  DA             da;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)vec,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Vector not associated with a DA");
  ierr = DAGetGhostCorners(da,0,0,0,&m,&n,0);CHKERRQ(ierr);

  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  mat  = mxCreateDoubleMatrix(m,n,mxREAL);
#else
  mat  = mxCreateDoubleMatrix(m,n,mxCOMPLEX);
#endif
  ierr = PetscMemcpy(mxGetPr(mat),array,n*m*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);
  engPutVariable((Engine *)mengine,obj->name,mat);
  
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif


#undef __FUNCT__  
#define __FUNCT__ "DACreateLocalVector"
/*@
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
PetscErrorCode PETSCDM_DLLEXPORT DACreateLocalVector(DA da,Vec* g)
{
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(g,2);
  ierr = VecCreateSeq(PETSC_COMM_SELF,da->nlocal,g);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*g,da->w);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*g,"DA",(PetscObject)da);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  if (da->w == 1  && da->dim == 2) {
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)*g,"PetscMatlabEnginePut_C","VecMatlabEnginePut_DA2d",VecMatlabEnginePut_DA2d);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetLocalVector"
/*@
   DMGetLocalVector - Gets a Seq PETSc vector that
   may be used with the DMXXX routines. This vector has spaces for the ghost values.

   Not Collective

   Input Parameter:
.  dm - the distributed array

   Output Parameter:
.  g - the local vector

   Level: beginner

   Note:
   The vector values are NOT initialized and may have garbage in them, so you may need
   to zero them.

   The output parameter, g, is a regular PETSc vector that should be returned with 
   DMRestoreLocalVector() DO NOT call VecDestroy() on it.

   VecStride*() operations can be useful when using DM with dof > 1

.keywords: distributed array, create, local, vector

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobal(), DMCreateLocalVector(), DMRestoreLocalVector(),
          VecStrideMax(), VecStrideMin(), VecStrideNorm()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetLocalVector(DM dm,Vec* g)
{
  PetscErrorCode ierr,i;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidPointer(g,2);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->localin[i]) {
      *g             = dm->localin[i];
      dm->localin[i] = PETSC_NULL;
      goto alldone;
    }
  }
  ierr = DMCreateLocalVector(dm,g);CHKERRQ(ierr);

  alldone:
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (!dm->localout[i]) {
      dm->localout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMRestoreLocalVector"
/*@
   DMRestoreLocalVector - Returns a Seq PETSc vector that
     obtained from DMGetLocalVector(). Do not use with vector obtained via
     DMCreateLocalVector().

   Not Collective

   Input Parameter:
+  dm - the distributed array
-  g - the local vector

   Level: beginner

.keywords: distributed array, create, local, vector

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobal(), DMCreateLocalVector(), DMGetLocalVector()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMRestoreLocalVector(DM dm,Vec* g)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidPointer(g,2);
  for (j=0; j<DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->localout[j]) {
      dm->localout[j] = PETSC_NULL;
      for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
        if (!dm->localin[i]) {
          dm->localin[i] = *g;
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
#define __FUNCT__ "DMGetGlobalVector"
/*@
   DMGetGlobalVector - Gets a MPI PETSc vector that
   may be used with the DMXXX routines.

   Collective on DM

   Input Parameter:
.  dm - the distributed array

   Output Parameter:
.  g - the global vector

   Level: beginner

   Note:
   The vector values are NOT initialized and may have garbage in them, so you may need
   to zero them.

   The output parameter, g, is a regular PETSc vector that should be returned with 
   DMRestoreGlobalVector() DO NOT call VecDestroy() on it.

   VecStride*() operations can be useful when using DM with dof > 1

.keywords: distributed array, create, Global, vector

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobal(), DMCreateLocalVector(), DMRestoreLocalVector()
          VecStrideMax(), VecStrideMin(), VecStrideNorm()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMGetGlobalVector(DM dm,Vec* g)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidPointer(g,2);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->globalin[i]) {
      *g             = dm->globalin[i];
      dm->globalin[i] = PETSC_NULL;
      goto alldone;
    }
  }
  ierr = DMCreateGlobalVector(dm,g);CHKERRQ(ierr);

  alldone:
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (!dm->globalout[i]) {
      dm->globalout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMRestoreGlobalVector"
/*@
   DMRestoreGlobalVector - Returns a Seq PETSc vector that
     obtained from DMGetGlobalVector(). Do not use with vector obtained via
     DMCreateGlobalVector().

   Not Collective

   Input Parameter:
+  dm - the distributed array
-  g - the global vector

   Level: beginner

.keywords: distributed array, create, global, vector

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DMGlobalToGlobalBegin(),
          DMGlobalToGlobalEnd(), DMGlobalToGlobal(), DMCreateLocalVector(), DMGetGlobalVector()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMRestoreGlobalVector(DM dm,Vec* g)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidPointer(g,2);
  for (j=0; j<DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->globalout[j]) {
      dm->globalout[j] = PETSC_NULL;
      for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
        if (!dm->globalin[i]) {
          dm->globalin[i] = *g;
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

EXTERN_C_BEGIN
#include "adic/ad_utils.h"
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DAGetAdicArray"
/*@C
     DAGetAdicArray - Gets an array of derivative types for a DA
          
    Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch

    Output Parameters:
+    ptr - array data structured to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly (may be null)
-    tdof - total number of degrees of freedom represented in array_start (may be null)

     Notes:
       The vector values are NOT initialized and may have garbage in them, so you may need
       to zero them.

       Returns the same type of object as the DAVecGetArray() except its elements are 
           derivative types instead of PetscScalars

     Level: advanced

.seealso: DARestoreAdicArray()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetAdicArray(DA da,PetscTruth ghosted,void **iptr,void **array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,deriv_type_size,xs,ys,xm,ym,zs,zm,itdof;
  char           *iarray_start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->adarrayghostedin[i]) {
        *iptr                   = da->adarrayghostedin[i];
        iarray_start            = (char*)da->adstartghostedin[i];
        itdof                   = da->ghostedtdof;
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
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->adarrayin[i]) {
        *iptr            = da->adarrayin[i];
        iarray_start     = (char*)da->adstartin[i];
        itdof            = da->tdof;
        da->adarrayin[i] = PETSC_NULL;
        da->adstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = da->xs;
    ys = da->ys;
    zs = da->zs;
    xm = da->xe-da->xs;
    ym = da->ye-da->ys;
    zm = da->ze-da->zs;
  }
  deriv_type_size = PetscADGetDerivTypeSize();

  switch (da->dim) {
    case 1: {
      void *ptr;
      itdof = xm;

      ierr  = PetscMalloc(xm*deriv_type_size,&iarray_start);CHKERRQ(ierr);

      ptr   = (void*)(iarray_start - xs*deriv_type_size);
      *iptr = (void*)ptr; 
      break;}
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*deriv_type_size,&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*deriv_type_size - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + deriv_type_size*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    case 3: {
      void ***ptr,**bptr;
      itdof = xm*ym*zm;

      ierr  = PetscMalloc((zm+1)*sizeof(void **)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*deriv_type_size,&iarray_start);CHKERRQ(ierr);

      ptr  = (void***)(iarray_start + xm*ym*zm*deriv_type_size - zs*sizeof(void*));
      bptr = (void**)(iarray_start + xm*ym*zm*deriv_type_size + zm*sizeof(void**));

      for(i=zs;i<zs+zm;i++) {
        ptr[i] = bptr + ((i-zs)*ym - ys);
      }
      for (i=zs; i<zs+zm; i++) {
        for (j=ys; j<ys+ym; j++) {
          ptr[i][j] = iarray_start + deriv_type_size*(xm*ym*(i-zs) + xm*(j-ys) - xs);
        }
      }

      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(PETSC_ERR_SUP,"Dimension %D not supported",da->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->adarrayghostedout[i]) {
        da->adarrayghostedout[i] = *iptr ;
        da->adstartghostedout[i] = iarray_start;
        da->ghostedtdof          = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->adarrayout[i]) {
        da->adarrayout[i] = *iptr ;
        da->adstartout[i] = iarray_start;
        da->tdof          = itdof;
        break;
      }
    }
  }
  if (i == DA_MAX_AD_ARRAYS+1) SETERRQ(PETSC_ERR_SUP,"Too many DA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DARestoreAdicArray"
/*@C
     DARestoreAdicArray - Restores an array of derivative types for a DA
          
    Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch

    Output Parameters:
+    ptr - array data structured to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly
-    tdof - total number of degrees of freedom represented in array_start

     Level: advanced

.seealso: DAGetAdicArray()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DARestoreAdicArray(DA da,PetscTruth ghosted,void **iptr,void **array_start,PetscInt *tdof)
{
  PetscInt  i;
  void      *iarray_start = 0;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->adarrayghostedout[i] == *iptr) {
        iarray_start             = da->adstartghostedout[i];
        da->adarrayghostedout[i] = PETSC_NULL;
        da->adstartghostedout[i] = PETSC_NULL;
        break;
      }
    }
    if (!iarray_start) SETERRQ(PETSC_ERR_ARG_WRONG,"Could not find array in checkout list");
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->adarrayghostedin[i]){
        da->adarrayghostedin[i] = *iptr;
        da->adstartghostedin[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->adarrayout[i] == *iptr) {
        iarray_start      = da->adstartout[i];
        da->adarrayout[i] = PETSC_NULL;
        da->adstartout[i] = PETSC_NULL;
        break;
      }
    }
    if (!iarray_start) SETERRQ(PETSC_ERR_ARG_WRONG,"Could not find array in checkout list");
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->adarrayin[i]){
        da->adarrayin[i]   = *iptr;
        da->adstartin[i]   = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ad_DAGetArray"
PetscErrorCode PETSCDM_DLLEXPORT ad_DAGetArray(DA da,PetscTruth ghosted,void **iptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetAdicArray(da,ghosted,iptr,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ad_DARestoreArray"
PetscErrorCode PETSCDM_DLLEXPORT ad_DARestoreArray(DA da,PetscTruth ghosted,void **iptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DARestoreAdicArray(da,ghosted,iptr,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "DAGetArray"
/*@C
     DAGetArray - Gets a work array for a DA
          
    Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch

    Output Parameters:
.    ptr - array data structured

    Note:  The vector values are NOT initialized and may have garbage in them, so you may need
           to zero them.

  Level: advanced

.seealso: DARestoreArray(), DAGetAdicArray()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetArray(DA da,PetscTruth ghosted,void **iptr)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,zs,zm;
  char           *iarray_start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ghosted) {
    for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
      if (da->arrayghostedin[i]) {
        *iptr                 = da->arrayghostedin[i];
        iarray_start          = (char*)da->startghostedin[i];
        da->arrayghostedin[i] = PETSC_NULL;
        da->startghostedin[i] = PETSC_NULL;
        
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
    for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
      if (da->arrayin[i]) {
        *iptr          = da->arrayin[i];
        iarray_start   = (char*)da->startin[i];
        da->arrayin[i] = PETSC_NULL;
        da->startin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = da->xs;
    ys = da->ys;
    zs = da->zs;
    xm = da->xe-da->xs;
    ym = da->ye-da->ys;
    zm = da->ze-da->zs;
  }

  switch (da->dim) {
    case 1: {
      void *ptr;

      ierr  = PetscMalloc(xm*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr   = (void*)(iarray_start - xs*sizeof(PetscScalar));
      *iptr = (void*)ptr; 
      break;}
    case 2: {
      void **ptr;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    case 3: {
      void ***ptr,**bptr;

      ierr  = PetscMalloc((zm+1)*sizeof(void **)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void***)(iarray_start + xm*ym*zm*sizeof(PetscScalar) - zs*sizeof(void*));
      bptr = (void**)(iarray_start + xm*ym*zm*sizeof(PetscScalar) + zm*sizeof(void**));
      for(i=zs;i<zs+zm;i++) {
        ptr[i] = bptr + ((i-zs)*ym - ys);
      }
      for (i=zs; i<zs+zm; i++) {
        for (j=ys; j<ys+ym; j++) {
          ptr[i][j] = iarray_start + sizeof(PetscScalar)*(xm*ym*(i-zs) + xm*(j-ys) - xs);
        }
      }

      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(PETSC_ERR_SUP,"Dimension %D not supported",da->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
      if (!da->arrayghostedout[i]) {
        da->arrayghostedout[i] = *iptr ;
        da->startghostedout[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
      if (!da->arrayout[i]) {
        da->arrayout[i] = *iptr ;
        da->startout[i] = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DARestoreArray"
/*@C
     DARestoreArray - Restores an array of derivative types for a DA
          
    Input Parameter:
+    da - information about my local patch
.    ghosted - do you want arrays for the ghosted or nonghosted patch
-    ptr - array data structured to be passed to ad_FormFunctionLocal()

     Level: advanced

.seealso: DAGetArray(), DAGetAdicArray()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DARestoreArray(DA da,PetscTruth ghosted,void **iptr)
{
  PetscInt  i;
  void      *iarray_start = 0;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ghosted) {
    for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
      if (da->arrayghostedout[i] == *iptr) {
        iarray_start           = da->startghostedout[i];
        da->arrayghostedout[i] = PETSC_NULL;
        da->startghostedout[i] = PETSC_NULL;
        break;
      }
    }
    for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
      if (!da->arrayghostedin[i]){
        da->arrayghostedin[i] = *iptr;
        da->startghostedin[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
      if (da->arrayout[i] == *iptr) {
        iarray_start    = da->startout[i];
        da->arrayout[i] = PETSC_NULL;
        da->startout[i] = PETSC_NULL;
        break;
      }
    }
    for (i=0; i<DA_MAX_WORK_ARRAYS; i++) {
      if (!da->arrayin[i]){
        da->arrayin[i]  = *iptr;
        da->startin[i]  = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAGetAdicMFArray"
/*@C
     DAGetAdicMFArray - Gets an array of derivative types for a DA for matrix-free ADIC.
          
     Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch?

     Output Parameters:
+    iptr - array data structured to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly (may be null)
-    tdof - total number of degrees of freedom represented in array_start (may be null)

     Notes: 
     The vector values are NOT initialized and may have garbage in them, so you may need
     to zero them.

     This routine returns the same type of object as the DAVecGetArray(), except its
     elements are derivative types instead of PetscScalars.

     Level: advanced

.seealso: DARestoreAdicMFArray(), DAGetArray(), DAGetAdicArray()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetAdicMFArray(DA da,PetscTruth ghosted,void **iptr,void **array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,zs,zm,itdof = 0;
  char           *iarray_start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayghostedin[i]) {
        *iptr                     = da->admfarrayghostedin[i];
        iarray_start              = (char*)da->admfstartghostedin[i];
        itdof                     = da->ghostedtdof;
        da->admfarrayghostedin[i] = PETSC_NULL;
        da->admfstartghostedin[i] = PETSC_NULL;
        
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
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayin[i]) {
        *iptr              = da->admfarrayin[i];
        iarray_start       = (char*)da->admfstartin[i];
        itdof              = da->tdof;
        da->admfarrayin[i] = PETSC_NULL;
        da->admfstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = da->xs;
    ys = da->ys;
    zs = da->zs;
    xm = da->xe-da->xs;
    ym = da->ye-da->ys;
    zm = da->ze-da->zs;
  }

  switch (da->dim) {
    case 1: {
      void *ptr;
      itdof = xm;

      ierr  = PetscMalloc(xm*2*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr   = (void*)(iarray_start - xs*2*sizeof(PetscScalar));
      *iptr = (void*)ptr; 
      break;}
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*2*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*2*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + 2*sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    case 3: {
      void ***ptr,**bptr;
      itdof = xm*ym*zm;

      ierr  = PetscMalloc((zm+1)*sizeof(void **)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*2*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void***)(iarray_start + xm*ym*zm*2*sizeof(PetscScalar) - zs*sizeof(void*));
      bptr = (void**)(iarray_start + xm*ym*zm*2*sizeof(PetscScalar) + zm*sizeof(void**));
      for(i=zs;i<zs+zm;i++) {
        ptr[i] = bptr + ((i-zs)*ym* - ys)*sizeof(void*);
      }
      for (i=zs; i<zs+zm; i++) {
        for (j=ys; j<ys+ym; j++) {
          ptr[i][j] = iarray_start + 2*sizeof(PetscScalar)*(xm*ym*(i-zs) + xm*(j-ys) - xs);
        }
      }

      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(PETSC_ERR_SUP,"Dimension %D not supported",da->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayghostedout[i]) {
        da->admfarrayghostedout[i] = *iptr ;
        da->admfstartghostedout[i] = iarray_start;
        da->ghostedtdof            = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayout[i]) {
        da->admfarrayout[i] = *iptr ;
        da->admfstartout[i] = iarray_start;
        da->tdof            = itdof;
        break;
      }
    }
  }
  if (i == DA_MAX_AD_ARRAYS+1) SETERRQ(PETSC_ERR_ARG_WRONG,"Too many DA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAGetAdicMFArray4"
PetscErrorCode PETSCDM_DLLEXPORT DAGetAdicMFArray4(DA da,PetscTruth ghosted,void **iptr,void **array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,zs,zm,itdof = 0;
  char           *iarray_start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayghostedin[i]) {
        *iptr                     = da->admfarrayghostedin[i];
        iarray_start              = (char*)da->admfstartghostedin[i];
        itdof                     = da->ghostedtdof;
        da->admfarrayghostedin[i] = PETSC_NULL;
        da->admfstartghostedin[i] = PETSC_NULL;
        
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
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayin[i]) {
        *iptr              = da->admfarrayin[i];
        iarray_start       = (char*)da->admfstartin[i];
        itdof              = da->tdof;
        da->admfarrayin[i] = PETSC_NULL;
        da->admfstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = da->xs;
    ys = da->ys;
    zs = da->zs;
    xm = da->xe-da->xs;
    ym = da->ye-da->ys;
    zm = da->ze-da->zs;
  }

  switch (da->dim) {
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*5*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*5*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + 5*sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(PETSC_ERR_SUP,"Dimension %D not supported",da->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayghostedout[i]) {
        da->admfarrayghostedout[i] = *iptr ;
        da->admfstartghostedout[i] = iarray_start;
        da->ghostedtdof            = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayout[i]) {
        da->admfarrayout[i] = *iptr ;
        da->admfstartout[i] = iarray_start;
        da->tdof            = itdof;
        break;
      }
    }
  }
  if (i == DA_MAX_AD_ARRAYS+1) SETERRQ(PETSC_ERR_ARG_WRONG,"Too many DA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAGetAdicMFArray9"
PetscErrorCode PETSCDM_DLLEXPORT DAGetAdicMFArray9(DA da,PetscTruth ghosted,void **iptr,void **array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,zs,zm,itdof = 0;
  char           *iarray_start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayghostedin[i]) {
        *iptr                     = da->admfarrayghostedin[i];
        iarray_start              = (char*)da->admfstartghostedin[i];
        itdof                     = da->ghostedtdof;
        da->admfarrayghostedin[i] = PETSC_NULL;
        da->admfstartghostedin[i] = PETSC_NULL;
        
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
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayin[i]) {
        *iptr              = da->admfarrayin[i];
        iarray_start       = (char*)da->admfstartin[i];
        itdof              = da->tdof;
        da->admfarrayin[i] = PETSC_NULL;
        da->admfstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = da->xs;
    ys = da->ys;
    zs = da->zs;
    xm = da->xe-da->xs;
    ym = da->ye-da->ys;
    zm = da->ze-da->zs;
  }

  switch (da->dim) {
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*10*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*10*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + 10*sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(PETSC_ERR_SUP,"Dimension %D not supported",da->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayghostedout[i]) {
        da->admfarrayghostedout[i] = *iptr ;
        da->admfstartghostedout[i] = iarray_start;
        da->ghostedtdof            = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayout[i]) {
        da->admfarrayout[i] = *iptr ;
        da->admfstartout[i] = iarray_start;
        da->tdof            = itdof;
        break;
      }
    }
  }
  if (i == DA_MAX_AD_ARRAYS+1) SETERRQ(PETSC_ERR_ARG_WRONG,"Too many DA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAGetAdicMFArrayb"
/*@C
     DAGetAdicMFArrayb - Gets an array of derivative types for a DA for matrix-free ADIC.
          
     Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch?

     Output Parameters:
+    iptr - array data structured to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly (may be null)
-    tdof - total number of degrees of freedom represented in array_start (may be null)

     Notes: 
     The vector values are NOT initialized and may have garbage in them, so you may need
     to zero them.

     This routine returns the same type of object as the DAVecGetArray(), except its
     elements are derivative types instead of PetscScalars.

     Level: advanced

.seealso: DARestoreAdicMFArray(), DAGetArray(), DAGetAdicArray()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetAdicMFArrayb(DA da,PetscTruth ghosted,void **iptr,void **array_start,PetscInt *tdof)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,zs,zm,itdof = 0;
  char           *iarray_start;
  PetscInt       bs = da->w,bs1 = bs+1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayghostedin[i]) {
        *iptr                     = da->admfarrayghostedin[i];
        iarray_start              = (char*)da->admfstartghostedin[i];
        itdof                     = da->ghostedtdof;
        da->admfarrayghostedin[i] = PETSC_NULL;
        da->admfstartghostedin[i] = PETSC_NULL;
        
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
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayin[i]) {
        *iptr              = da->admfarrayin[i];
        iarray_start       = (char*)da->admfstartin[i];
        itdof              = da->tdof;
        da->admfarrayin[i] = PETSC_NULL;
        da->admfstartin[i] = PETSC_NULL;
        
        goto done;
      }
    }
    xs = da->xs;
    ys = da->ys;
    zs = da->zs;
    xm = da->xe-da->xs;
    ym = da->ye-da->ys;
    zm = da->ze-da->zs;
  }

  switch (da->dim) {
    case 1: {
      void *ptr;
      itdof = xm;

      ierr  = PetscMalloc(xm*bs1*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr   = (void*)(iarray_start - xs*bs1*sizeof(PetscScalar));
      *iptr = (void*)ptr; 
      break;}
    case 2: {
      void **ptr;
      itdof = xm*ym;

      ierr  = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*bs1*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void**)(iarray_start + xm*ym*bs1*sizeof(PetscScalar) - ys*sizeof(void*));
      for(j=ys;j<ys+ym;j++) {
        ptr[j] = iarray_start + bs1*sizeof(PetscScalar)*(xm*(j-ys) - xs);
      }
      *iptr = (void*)ptr; 
      break;}
    case 3: {
      void ***ptr,**bptr;
      itdof = xm*ym*zm;

      ierr  = PetscMalloc((zm+1)*sizeof(void **)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*bs1*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

      ptr  = (void***)(iarray_start + xm*ym*zm*2*sizeof(PetscScalar) - zs*sizeof(void*));
      bptr = (void**)(iarray_start + xm*ym*zm*2*sizeof(PetscScalar) + zm*sizeof(void**));
      for(i=zs;i<zs+zm;i++) {
        ptr[i] = bptr + ((i-zs)*ym* - ys)*sizeof(void*);
      }
      for (i=zs; i<zs+zm; i++) {
        for (j=ys; j<ys+ym; j++) {
          ptr[i][j] = iarray_start + bs1*sizeof(PetscScalar)*(xm*ym*(i-zs) + xm*(j-ys) - xs);
        }
      }

      *iptr = (void*)ptr; 
      break;}
    default:
      SETERRQ1(PETSC_ERR_SUP,"Dimension %D not supported",da->dim);
  }

  done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayghostedout[i]) {
        da->admfarrayghostedout[i] = *iptr ;
        da->admfstartghostedout[i] = iarray_start;
        da->ghostedtdof            = itdof;
        break;
      }
    }
  } else {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayout[i]) {
        da->admfarrayout[i] = *iptr ;
        da->admfstartout[i] = iarray_start;
        da->tdof            = itdof;
        break;
      }
    }
  }
  if (i == DA_MAX_AD_ARRAYS+1) SETERRQ(PETSC_ERR_ARG_WRONG,"Too many DA ADIC arrays obtained");
  if (tdof)        *tdof = itdof;
  if (array_start) *array_start = iarray_start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DARestoreAdicMFArray"
/*@C
     DARestoreAdicMFArray - Restores an array of derivative types for a DA.
          
     Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch?

     Output Parameters:
+    ptr - array data structure to be passed to ad_FormFunctionLocal()
.    array_start - actual start of 1d array of all values that adiC can access directly
-    tdof - total number of degrees of freedom represented in array_start

     Level: advanced

.seealso: DAGetAdicArray()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DARestoreAdicMFArray(DA da,PetscTruth ghosted,void **iptr,void **array_start,PetscInt *tdof)
{
  PetscInt  i;
  void      *iarray_start = 0;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ghosted) {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayghostedout[i] == *iptr) {
        iarray_start               = da->admfstartghostedout[i];
        da->admfarrayghostedout[i] = PETSC_NULL;
        da->admfstartghostedout[i] = PETSC_NULL;
        break;
      }
    }
    if (!iarray_start) SETERRQ(PETSC_ERR_ARG_WRONG,"Could not find array in checkout list");
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayghostedin[i]){
        da->admfarrayghostedin[i] = *iptr;
        da->admfstartghostedin[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (da->admfarrayout[i] == *iptr) {
        iarray_start        = da->admfstartout[i];
        da->admfarrayout[i] = PETSC_NULL;
        da->admfstartout[i] = PETSC_NULL;
        break;
      }
    }
    if (!iarray_start) SETERRQ(PETSC_ERR_ARG_WRONG,"Could not find array in checkout list");
    for (i=0; i<DA_MAX_AD_ARRAYS; i++) {
      if (!da->admfarrayin[i]){
        da->admfarrayin[i] = *iptr;
        da->admfstartin[i] = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "admf_DAGetArray"
PetscErrorCode PETSCDM_DLLEXPORT admf_DAGetArray(DA da,PetscTruth ghosted,void **iptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetAdicMFArray(da,ghosted,iptr,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "admf_DARestoreArray"
PetscErrorCode PETSCDM_DLLEXPORT admf_DARestoreArray(DA da,PetscTruth ghosted,void **iptr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DARestoreAdicMFArray(da,ghosted,iptr,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*M
  DAGetLocalVector - same as DMGetLocalVector

  Synopsis:
  PetscErrorCode DAGetLocalVector(DM dm,Vec* g)

  Not Collective

  Level: beginner
M*/


/*M
  DARestoreLocalVector - same as DMRestoreLocalVector

  Synopsis:
  PetscErrorCode DARestoreLocalVector(DM dm,Vec* g)

  Not Collective

  Level: beginner
M*/


/*M
  DAGetGlobalVector - same as DMGetGlobalVector

  Synopsis:
  PetscErrorCode  DAGetGlobalVector(DM dm,Vec* g)

  Collective on DM

  Level: beginner
M*/


/*M
  DARestoreGlobalVector - same as DMRestoreGlobalVector

  Synopsis:
  PetscErrorCode DARestoreGlobalVector(DM dm,Vec* g)

  Collective on DM

  Level: beginner
M*/
