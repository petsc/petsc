#define PETSCVEC_DLL

#include "petscvec.h"    /*I "petscvec.h" I*/

#include "engine.h"   /* Matlab include file */
#include "mex.h"      /* Matlab include file */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecMatlabEnginePut_Default"
PetscErrorCode PETSCVEC_DLLEXPORT VecMatlabEnginePut_Default(PetscObject obj,void *mengine)
{
  PetscErrorCode ierr;
  PetscInt       n;
  Vec            vec = (Vec)obj;
  PetscScalar    *array;
  mxArray        *mat;

  PetscFunctionBegin;
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  mat  = mxCreateDoubleMatrix(n,1,mxREAL);
#else
  mat  = mxCreateDoubleMatrix(n,1,mxCOMPLEX);
#endif
  ierr = PetscMemcpy(mxGetPr(mat),array,n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);
  engPutVariable((Engine *)mengine,obj->name,mat);
  
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecMatlabEngineGet_Default"
PetscErrorCode PETSCVEC_DLLEXPORT VecMatlabEngineGet_Default(PetscObject obj,void *mengine)
{
  PetscErrorCode ierr;
  PetscInt       n;
  Vec            vec = (Vec)obj;
  PetscScalar    *array;
  mxArray        *mat;

  PetscFunctionBegin;
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  mat  = engGetVariable((Engine *)mengine,obj->name);
  if (!mat) SETERRQ1(PETSC_ERR_LIB,"Unable to get object %s from matlab",obj->name);
  ierr = PetscMemcpy(array,mxGetPr(mat),n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END



