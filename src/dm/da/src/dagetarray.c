#define PETSCDM_DLL
 
#include "petscda.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DAVecGetArray"
/*@C
   DAVecGetArray - Returns a multiple dimension array that shares data with 
      the underlying vector and is indexed using the global dimensions.

   Not Collective

   Input Parameter:
+  da - the distributed array
-  vec - the vector, either a vector the same size as one obtained with 
         DACreateGlobalVector() or DACreateLocalVector()
   
   Output Parameter:
.  array - the array

   Notes:
    Call DAVecRestoreArray() once you have finished accessing the vector entries.

    In C, the indexing is "backwards" from what expects: array[k][j][i] NOT array[i][j][k]!

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DAGetCorners(), VecGetArray(), VecRestoreArray(), DAVecRestoreArray(), DAVecRestoreArrayDOF()
          DAVecGetarrayDOF()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAVecGetArray(DA da,Vec vec,void *array)
{
  PetscErrorCode ierr;
  PetscInt       xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

  PetscFunctionBegin;
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);

  /* Handle case where user passes in global vector as opposed to local */
  ierr = VecGetLocalSize(vec,&N);CHKERRQ(ierr);
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  } else if (N != gxm*gym*gzm*dof) {
    SETERRQ3(PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DA local sizes %D %D\n",N,xm*ym*zm*dof,gxm*gym*gzm*dof);
  }

  if (dim == 1) {
    ierr = VecGetArray1d(vec,gxm*dof,gxs*dof,(PetscScalar **)array);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = VecGetArray2d(vec,gym,gxm*dof,gys,gxs*dof,(PetscScalar***)array);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = VecGetArray3d(vec,gzm,gym,gxm*dof,gzs,gys,gxs*dof,(PetscScalar****)array);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_ARG_CORRUPT,"DA dimension not 1, 2, or 3, it is %D\n",dim);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAVecRestoreArray"
/*@
   DAVecRestoreArray - Restores a multiple dimension array obtained with DAVecGetArray()

   Not Collective

   Input Parameter:
+  da - the distributed array
.  vec - the vector, either a vector the same size as one obtained with 
         DACreateGlobalVector() or DACreateLocalVector()
-  array - the array

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DAGetCorners(), VecGetArray(), VecRestoreArray(), DAVecGetArray()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAVecRestoreArray(DA da,Vec vec,void *array)
{
  PetscErrorCode ierr;
  PetscInt       xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

  PetscFunctionBegin;
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);

  /* Handle case where user passes in global vector as opposed to local */
  ierr = VecGetLocalSize(vec,&N);CHKERRQ(ierr);
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  } else if (N != gxm*gym*gzm*dof) {
    SETERRQ3(PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DA local sizes %D %D\n",N,xm*ym*zm*dof,gxm*gym*gzm*dof);
  }

  if (dim == 1) {
    ierr = VecRestoreArray1d(vec,gxm*dof,gxs*dof,(PetscScalar **)array);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = VecRestoreArray2d(vec,gym,gxm*dof,gys,gxs*dof,(PetscScalar***)array);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = VecRestoreArray3d(vec,gzm,gym,gxm*dof,gzs,gys,gxs*dof,(PetscScalar****)array);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_ARG_CORRUPT,"DA dimension not 1, 2, or 3, it is %D\n",dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAVecGetArrayDOF"
/*@C
   DAVecGetArrayDOF - Returns a multiple dimension array that shares data with 
      the underlying vector and is indexed using the global dimensions.

   Not Collective

   Input Parameter:
+  da - the distributed array
-  vec - the vector, either a vector the same size as one obtained with 
         DACreateGlobalVector() or DACreateLocalVector()
   
   Output Parameter:
.  array - the array

   Notes:
    Call DAVecRestoreArrayDOF() once you have finished accessing the vector entries.

    In C, the indexing is "backwards" from what expects: array[k][j][i][DOF] NOT array[i][j][k][DOF]!

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DAGetCorners(), VecGetArray(), VecRestoreArray(), DAVecRestoreArray(), DAVecGetArray(), DAVecRestoreArrayDOF()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAVecGetArrayDOF(DA da,Vec vec,void *array)
{
  PetscErrorCode ierr;
  PetscInt       xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

  PetscFunctionBegin;
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);

  /* Handle case where user passes in global vector as opposed to local */
  ierr = VecGetLocalSize(vec,&N);CHKERRQ(ierr);
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  } else if (N != gxm*gym*gzm*dof) {
    SETERRQ3(PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DA local sizes %D %D\n",N,xm*ym*zm*dof,gxm*gym*gzm*dof);
  }

  if (dim == 1) {
    ierr = VecGetArray2d(vec,gxm,dof,gxs,0,(PetscScalar ***)array);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = VecGetArray3d(vec,gym,gxm,dof,gys,gxs,0,(PetscScalar****)array);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = VecGetArray4d(vec,gzm,gym,gxm,dof,gzs,gys,gxs,0,(PetscScalar*****)array);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_ARG_CORRUPT,"DA dimension not 1, 2, or 3, it is %D\n",dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAVecRestoreArrayDOF"
/*@
   DAVecRestoreArrayDOF - Restores a multiple dimension array obtained with DAVecGetArrayDOF()

   Not Collective

   Input Parameter:
+  da - the distributed array
.  vec - the vector, either a vector the same size as one obtained with 
         DACreateGlobalVector() or DACreateLocalVector()
-  array - the array

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DAGetCorners(), VecGetArray(), VecRestoreArray(), DAVecGetArray(), DAVecGetArrayDOF(), DAVecRestoreArrayDOF()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAVecRestoreArrayDOF(DA da,Vec vec,void *array)
{
  PetscErrorCode ierr;
  PetscInt       xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

  PetscFunctionBegin;
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);

  /* Handle case where user passes in global vector as opposed to local */
  ierr = VecGetLocalSize(vec,&N);CHKERRQ(ierr);
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  }

  if (dim == 1) {
    ierr = VecRestoreArray2d(vec,gxm,dof,gxs,0,(PetscScalar***)array);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = VecRestoreArray3d(vec,gym,gxm,dof,gys,gxs,0,(PetscScalar****)array);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = VecRestoreArray4d(vec,gzm,gym,gxm,dof,gzs,gys,gxs,0,(PetscScalar*****)array);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_ARG_CORRUPT,"DA dimension not 1, 2, or 3, it is %D\n",dim);
  }
  PetscFunctionReturn(0);
}













