
#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMDAVecGetArray"
/*@C
   DMDAVecGetArray - Returns a multiple dimension array that shares data with
      the underlying vector and is indexed using the global dimensions.

   Not Collective

   Input Parameter:
+  da - the distributed array
-  vec - the vector, either a vector the same size as one obtained with
         DMCreateGlobalVector() or DMCreateLocalVector()

   Output Parameter:
.  array - the array

   Notes:
    Call DMDAVecRestoreArray() once you have finished accessing the vector entries.

    In C, the indexing is "backwards" from what expects: array[k][j][i] NOT array[i][j][k]!

    If vec is a local vector (obtained with DMCreateLocalVector() etc) then they ghost point locations are accessable. If it is
    a global vector then the ghost points are not accessable. Of course with the local vector you will have had to do the

    appropriate DMLocalToGlobalBegin() and DMLocalToGlobalEnd() to have correct values in the ghost locations.

  Fortran Notes: From Fortran use DMDAVecGetArrayF90() and pass for the array type PetscScalar,pointer :: array(:,...,:) of the appropriate
       dimension. For a DMDA created with a dof of 1 use the dimension of the DMDA, for a DMDA created with a dof greater than 1 use one more than the
       dimension of the DMDA. The order of the indices is array(xs:xs+xm-1,ys:ys+ym-1,zs:zs+zm-1) (when dof is 1) otherwise
       array(0:dof-1,xs:xs+xm-1,ys:ys+ym-1,zs:zs+zm-1) where the values are obtained from
       DMDAGetCorners() for a global array or DMDAGetGhostCorners() for a local array. Include finclude/petscdmda.h90 to access this routine.

  Due to bugs in the compiler DMDAVecGetArrayF90() does not work with gfortran versions before 4.5

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DMDAGetGhostCorners(), DMDAGetCorners(), VecGetArray(), VecRestoreArray(), DMDAVecRestoreArray(), DMDAVecRestoreArrayDOF()
          DMDAVecGetArrayDOF()
@*/
PetscErrorCode  DMDAVecGetArray(DM da,Vec vec,void *array)
{
  PetscErrorCode ierr;
  PetscInt       xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  if (da->defaultSection) {
    ierr = VecGetArray(vec,(PetscScalar**)array);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);

  /* Handle case where user passes in global vector as opposed to local */
  ierr = VecGetLocalSize(vec,&N);CHKERRQ(ierr);
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  } else if (N != gxm*gym*gzm*dof) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMDA local sizes %D %D\n",N,xm*ym*zm*dof,gxm*gym*gzm*dof);

  if (dim == 1) {
    ierr = VecGetArray1d(vec,gxm*dof,gxs*dof,(PetscScalar **)array);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = VecGetArray2d(vec,gym,gxm*dof,gys,gxs*dof,(PetscScalar***)array);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = VecGetArray3d(vec,gzm,gym,gxm*dof,gzs,gys,gxs*dof,(PetscScalar****)array);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"DMDA dimension not 1, 2, or 3, it is %D\n",dim);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAVecRestoreArray"
/*@
   DMDAVecRestoreArray - Restores a multiple dimension array obtained with DMDAVecGetArray()

   Not Collective

   Input Parameter:
+  da - the distributed array
.  vec - the vector, either a vector the same size as one obtained with
         DMCreateGlobalVector() or DMCreateLocalVector()
-  array - the array

  Level: intermediate

  Fortran Notes: From Fortran use DMDAVecRestoreArrayF90()

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DMDAGetGhostCorners(), DMDAGetCorners(), VecGetArray(), VecRestoreArray(), DMDAVecGetArray()
@*/
PetscErrorCode  DMDAVecRestoreArray(DM da,Vec vec,void *array)
{
  PetscErrorCode ierr;
  PetscInt       xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  if (da->defaultSection) {
    ierr = VecRestoreArray(vec,(PetscScalar**)array);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);

  /* Handle case where user passes in global vector as opposed to local */
  ierr = VecGetLocalSize(vec,&N);CHKERRQ(ierr);
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  } else if (N != gxm*gym*gzm*dof) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMDA local sizes %D %D\n",N,xm*ym*zm*dof,gxm*gym*gzm*dof);

  if (dim == 1) {
    ierr = VecRestoreArray1d(vec,gxm*dof,gxs*dof,(PetscScalar **)array);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = VecRestoreArray2d(vec,gym,gxm*dof,gys,gxs*dof,(PetscScalar***)array);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = VecRestoreArray3d(vec,gzm,gym,gxm*dof,gzs,gys,gxs*dof,(PetscScalar****)array);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"DMDA dimension not 1, 2, or 3, it is %D\n",dim);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAVecGetArrayDOF"
/*@C
   DMDAVecGetArrayDOF - Returns a multiple dimension array that shares data with
      the underlying vector and is indexed using the global dimensions.

   Not Collective

   Input Parameter:
+  da - the distributed array
-  vec - the vector, either a vector the same size as one obtained with
         DMCreateGlobalVector() or DMCreateLocalVector()

   Output Parameter:
.  array - the array

   Notes:
    Call DMDAVecRestoreArrayDOF() once you have finished accessing the vector entries.

    In C, the indexing is "backwards" from what expects: array[k][j][i][DOF] NOT array[i][j][k][DOF]!

    In Fortran 90 you do not need a version of DMDAVecRestoreArrayDOF() just use  DMDAVecRestoreArrayF90() and declare your array with one higher dimension,
    see src/dm/examples/tutorials/ex11f90.F

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DMDAGetGhostCorners(), DMDAGetCorners(), VecGetArray(), VecRestoreArray(), DMDAVecRestoreArray(), DMDAVecGetArray(), DMDAVecRestoreArrayDOF()
@*/
PetscErrorCode  DMDAVecGetArrayDOF(DM da,Vec vec,void *array)
{
  PetscErrorCode ierr;
  PetscInt       xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);

  /* Handle case where user passes in global vector as opposed to local */
  ierr = VecGetLocalSize(vec,&N);CHKERRQ(ierr);
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  } else if (N != gxm*gym*gzm*dof) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMDA local sizes %D %D\n",N,xm*ym*zm*dof,gxm*gym*gzm*dof);

  if (dim == 1) {
    ierr = VecGetArray2d(vec,gxm,dof,gxs,0,(PetscScalar ***)array);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = VecGetArray3d(vec,gym,gxm,dof,gys,gxs,0,(PetscScalar****)array);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = VecGetArray4d(vec,gzm,gym,gxm,dof,gzs,gys,gxs,0,(PetscScalar*****)array);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"DMDA dimension not 1, 2, or 3, it is %D\n",dim);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAVecRestoreArrayDOF"
/*@
   DMDAVecRestoreArrayDOF - Restores a multiple dimension array obtained with DMDAVecGetArrayDOF()

   Not Collective

   Input Parameter:
+  da - the distributed array
.  vec - the vector, either a vector the same size as one obtained with
         DMCreateGlobalVector() or DMCreateLocalVector()
-  array - the array

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DMDAGetGhostCorners(), DMDAGetCorners(), VecGetArray(), VecRestoreArray(), DMDAVecGetArray(), DMDAVecGetArrayDOF(), DMDAVecRestoreArrayDOF()
@*/
PetscErrorCode  DMDAVecRestoreArrayDOF(DM da,Vec vec,void *array)
{
  PetscErrorCode ierr;
  PetscInt       xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);

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
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"DMDA dimension not 1, 2, or 3, it is %D\n",dim);
  PetscFunctionReturn(0);
}













