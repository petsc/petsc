
#include <petscvec.h>    /*I "petscvec.h" I*/
#include <petsc/private/petscimpl.h>

#include <engine.h>   /* MATLAB include file */
#include <mex.h>      /* MATLAB include file */

PETSC_EXTERN PetscErrorCode VecMatlabEnginePut_Default(PetscObject obj, void *mengine)
{
  PetscInt          n;
  Vec               vec = (Vec)obj;
  const PetscScalar *array;
  mxArray           *mat;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(vec,&array));
  CHKERRQ(VecGetLocalSize(vec,&n));
#if defined(PETSC_USE_COMPLEX)
  mat = mxCreateDoubleMatrix(n,1,mxCOMPLEX);
#else
  mat = mxCreateDoubleMatrix(n,1,mxREAL);
#endif
  CHKERRQ(PetscArraycpy(mxGetPr(mat),array,n));
  CHKERRQ(PetscObjectName(obj));
  engPutVariable((Engine*)mengine,obj->name,mat);

  CHKERRQ(VecRestoreArrayRead(vec,&array));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecMatlabEngineGet_Default(PetscObject obj, void *mengine)
{
  PetscInt     n;
  Vec          vec = (Vec)obj;
  PetscScalar *array;
  mxArray     *mat;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(vec,&array));
  CHKERRQ(VecGetLocalSize(vec,&n));
  mat = engGetVariable((Engine*)mengine,obj->name);
  PetscCheck(mat,PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to get object %s from matlab",obj->name);
  CHKERRQ(PetscArraycpy(array,mxGetPr(mat),n));
  CHKERRQ(VecRestoreArray(vec,&array));
  PetscFunctionReturn(0);
}
