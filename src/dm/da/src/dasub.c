#ifndef lint
static char vcid[] = "$Id: dacorn.c,v 1.4 1996/03/19 21:29:33 bsmith Exp $";
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
   or application statistics that require sums of data throughout
   a logical plane of grid points.

.keywords: distributed array, get, processor subset
@*/
int DAGetProcessorSubset(DA da,DADirection dir,int gp,MPI_Comm *comm)
{
  int xs,xm,xe,ys,ym,ye,zs,zm,ze;

  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = DAGetCorners(da,&xs,&xm,&ys,&ym,&zs,&zm); CHKERRQ(ierr);
  if (dir == DA_Z) {
    if (da->dim < 3) SETERRQ(1,"DAGetProcessorSubset:DA_Z invalid for DA dim < 3");
    ze = zs+zm;
    if (gp >= zs && gp < ze) 
  } else if (dir == DA_Y) {
    if (da->dim == 1) SETERRQ(1,"DAGetProcessorSubset:DA_Y invalid for DA dim = 1");
  } else if (dir == DA_X) {
  } else SETERRQ(1,"DAGetProcessorSubset:Invalid direction");

  return 0;
} 

