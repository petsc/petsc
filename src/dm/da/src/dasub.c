#ifndef lint
static char vcid[] = "$Id: dasub.c,v 1.3 1996/06/22 17:09:43 curfman Exp curfman $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "daimpl.h"    /*I   "da.h"   I*/

/*@
   DAGetProcessorSubset - Returns a communicator consisting only of the
   processors in a DA that own a particular global x, y, or z grid point
   (corresponding to a logical plane in a 3D grid or a line in a 2D grid).

   Input Parameters:
.  da - the distributed array
.  dir - Cartesian direction, either DA_X, DA_Y, or DA_Z
.  gp - global grid point number in this direction

   Output Parameters:
.  comm - new communicator

   Notes:
   This routine is particularly useful to compute boundary conditions
   or other application-specific calculations that require manipulating
   sets of data throughout a logical plane of grid points.

.keywords: distributed array, get, processor subset
@*/
int DAGetProcessorSubset(DA da,DADirection dir,int gp,MPI_Comm *comm)
{
  MPI_Group group, subgroup;
  int       ierr,i,ict,flag,size,*ranks,*owners,xs,xm,ys,ym,zs,zm;

  PetscValidHeaderSpecific(da,DA_COOKIE);
  flag = 0; 
  ierr = DAGetCorners(da,&xs,&xm,&ys,&ym,&zs,&zm); CHKERRQ(ierr);
  MPI_Comm_size(da->comm,&size);
  if (dir == DA_Z) {
    if (da->dim < 3) SETERRQ(1,"DAGetProcessorSubset:DA_Z invalid for DA dim < 3");
    if (gp < 0 || gp > da->P) SETERRQ(1,"DAGetProcessorSubset:invalid grid point");
    if (gp >= zs && gp < zs+zm) flag = 1;
  } else if (dir == DA_Y) {
    if (da->dim == 1) SETERRQ(1,"DAGetProcessorSubset:DA_Y invalid for DA dim = 1");
    if (gp < 0 || gp > da->N) SETERRQ(1,"DAGetProcessorSubset:invalid grid point");
    if (gp >= ys && gp < ys+ym) flag = 1;
  } else if (dir == DA_X) {
    if (gp < 0 || gp > da->M) SETERRQ(1,"DAGetProcessorSubset:invalid grid point");
    if (gp >= xs && gp < xs+xm) flag = 1;
  } else SETERRQ(1,"DAGetProcessorSubset:Invalid direction");

  owners = (int *)PetscMalloc(2*size*sizeof(int)); CHKPTRQ(owners);
  ranks = owners + size;
  MPI_Allgather(&flag,1,MPI_INT,owners,1,MPI_INT,da->comm);
  ict = 0;
  PLogInfo(da,"DAGetProcessorSubset: dim=%d, direction=%d, procs: ",da->dim,(int)dir);
  for (i=0; i<size; i++) {
    if (owners[i]) {
      ranks[ict] = i; ict++;
      PLogInfo(da,"%d ",i);
    }
  }
  PLogInfo(da,"\n");
  MPI_Comm_group(da->comm,&group);
  MPI_Group_incl(group,ict,ranks,&subgroup);
  MPI_Comm_create(da->comm,subgroup,comm);
  PetscFree(owners);
  return 0;
} 

/*@
   DAGetGridSubset - Returns the local grid indices for a subblock of a DA.

   Input Parameters:
.  da - the distributed array
.  gxs, gys, gzs - global starting grid points
.  gxe, gye, gze - global ending grid points

   Output Parameters:
.  lxs, lys, lzs - local starting grid points
.  lxe, lye, lze - local ending grid points

   Restrictions:
   gxs <= gxe, gys <= gye, gzs <= gze

   Notes:
   This routine is particularly useful to compute boundary conditions
   or other application-specific calculations that require manipulating
   sets of data throughout a subblock of a grid.

.keywords: distributed array, get, processor subset
@*/
int DAGetGridSubset(DA da,int gxs,int gxe,int gys,int gye,int gzs,int gze,
                    int *lxs,int *lxe,int *lys,int *lye,int *lzs,int *lze)
{
  int ierr,xs,xm,ys,ym,zs,zm;

  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = DAGetCorners(da,&xs,&xm,&ys,&ym,&zs,&zm); CHKERRQ(ierr);
  if (gxs == PETSC_DEFAULT || gxs == PETSC_DECIDE) gxs = 0;
  if (gxe == PETSC_DEFAULT || gxe == PETSC_DECIDE) gxe = da->M-1;
  if (gxs < 0 || gxe >= da->M || gxs > gxe)
    SETERRQ(1,"DAGetGridSubset:invalid grid point(s): 0 <= gxs <= gxe < M");
  if (lxs != PETSC_NULL) *lxs = PetscMax(gxs,xs);
  if (lxe != PETSC_NULL) *lxe = PetscMin(gxe,xs+xm);
  if (da->dim > 1) {
    if (gys == PETSC_DEFAULT || gys == PETSC_DECIDE) gys = 0;
    if (gye == PETSC_DEFAULT || gye == PETSC_DECIDE) gye = da->N-1;
    if (gys < 0 || gye >= da->N || gys > gye)
      SETERRQ(1,"DAGetGridSubset:invalid grid point(s): 0 <= gys <= gye < N");
    if (lys != PETSC_NULL) *lys = PetscMax(gys,ys);
    if (lye != PETSC_NULL) *lye = PetscMin(gye,ys+ym);
    if (da->dim > 2) {
      if (gzs == PETSC_DEFAULT || gzs == PETSC_DECIDE) gzs = 0;
      if (gze == PETSC_DEFAULT || gze == PETSC_DECIDE) gze = da->P-1;
      if (gzs < 0 || gze >= da->P || gzs > gze)
        SETERRQ(1,"DAGetGridSubset:invalid grid point(s): 0 <= gzs <= gze < P");
      if (lzs != PETSC_NULL) *lzs = PetscMax(gzs,zs);
      if (lze != PETSC_NULL) *lze = PetscMin(gze,zs+zm);
    }
    else {
      if (lze != PETSC_NULL) *lze = da->ze;
      if (lzs != PETSC_NULL) *lzs = da->zs;
    }
  }
  else {
    if (lye != PETSC_NULL) *lye = da->ye;
    if (lys != PETSC_NULL) *lys = da->ys;
  }
  return 0;
} 
