/*$Id: dagetarray.c,v 1.5 2000/04/12 04:26:20 bsmith Exp balay $*/
 
#include "petscda.h"    /*I   "petscda.h"   I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAVecGetArray"
/*@
   DAVecGetArray - Returns an multiple dimension array that shares data with 
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

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DAGetCorners(), VecGetArray(), VecRestoreArray(), DAVecRestoreArray()
@*/
int DAVecGetArray(DA da,Vec vec,void **array)
{
  int ierr;
  int xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

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
    SETERRQ3(1,1,"Vector local size %d is not compatible with DA local sizes %d %d\n",N,xm*ym*zm*dof,gxm*gym*gzm*dof);
  }

  if (dim == 1) {
    if (dof == 1) {
      ierr = VecGetArray1d(vec,gxm,gxs,(Scalar **)array);CHKERRQ(ierr);
    } else {
      ierr = VecGetArray2d(vec,gxm,dof,gxs,0,(Scalar***)array);CHKERRQ(ierr);
    }
  } else if (dim == 2) {
    if (dof == 1) {
      ierr = VecGetArray2d(vec,gym,gxm,gys,gxs,(Scalar***)array);CHKERRQ(ierr);
    } else {
      SETERRQ(1,1,"Not yet done");
    }
  } else if (dim == 3) {
    if (dof == 1) {
      SETERRQ(1,1,"Not yet done");
    } else {
      SETERRQ(1,1,"Not yet done");
    }
  } else {
    SETERRQ1(1,1,"DA dimension not 1, 2, or 3, it is %d\n",dim);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAVecRestoreArray"
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
int DAVecRestoreArray(DA da,Vec vec,void **array)
{
  int ierr;
  int xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

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
    if (dof == 1) {
      ierr = VecRestoreArray1d(vec,gxm,gxs,(Scalar **)array);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray2d(vec,gxm,dof,gxs,0,(Scalar***)array);CHKERRQ(ierr);
    }
  } else if (dim == 2) {
    if (dof == 1) {
      ierr = VecRestoreArray2d(vec,gym,gxm,gys,gxs,(Scalar***)array);CHKERRQ(ierr);
    } else {
      ;
    }
  } else if (dim == 3) {
    if (dof == 1) {
      ;
    } else {
      ;
    }
  } else {
    SETERRQ1(1,1,"DA dimension not 1, 2, or 3, it is %d\n",dim);
  }
  PetscFunctionReturn(0);
}
