
#include "petscvec.h"    /*I "petscvec.h" I*/

#if defined(PETSC_HAVE_MATLAB) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#include "engine.h"   /* Matlab include file */
#include "mex.h"      /* Matlab include file */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecMatlabEnginePut_Default"
int VecMatlabEnginePut_Default(PetscObject obj,void *mengine)
{
  int         ierr,n;
  Vec         vec = (Vec)obj;
  PetscScalar *array;
  mxArray     *mat;

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
int VecMatlabEngineGet_Default(PetscObject obj,void *mengine)
{
  int         ierr,n;
  Vec         vec = (Vec)obj;
  PetscScalar *array;
  mxArray     *mat;

  PetscFunctionBegin;
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  mat  = engGetVariable((Engine *)mengine,obj->name);
  if (!mat) SETERRQ1(1,"Unable to get object %s from matlab",obj->name);
  ierr = PetscMemcpy(array,mxGetPr(mat),n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif



