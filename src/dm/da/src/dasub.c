#ifndef lint
static char vcid[] = "$Id: dasub.c,v 1.6 1996/08/08 14:47:19 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

/*@C
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
