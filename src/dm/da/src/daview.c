#ifndef lint
static char vcid[] = "$Id: daview.c,v 1.9 1996/04/16 21:57:00 curfman Exp curfman $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "daimpl.h"    /*I   "da.h"   I*/

/*@
   DAView - Visualizes a distributed array object.

   Input Parameters:
.  da - the distributed array
.  ptr - an optional visualization context

   Notes:
   The available visualization contexts include
$     STDOUT_VIEWER_SELF - standard output (default)
$     STDOUT_VIEWER_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file
$    DrawOpenX() - output processor layout to an 
$         X window display

   Default Output Format:
$ (for 3d arrays):
$
$   Processor [proc] M  N  P  m  n  p  w  s
$   X range: xs xe, Y range: ys, ye, Z range: zs, ze
$
$   where
$      M,N,P - global dimension in each direction of the array
$      m,n,p - corresponding number of procs in each dimension 
$      w - number of degrees of freedom per node
$      s - stencil width
$      xs, xe - internal local starting/ending grid points
$               in x-direction, (augmented to handle multiple 
$               degrees of freedom per node)
$      ys, ye - local starting/ending grid points in y-direction
$      zs, ze - local starting/ending grid points in z-direction

   Options Database Key:
$  -da_view : call DAView() at the conclusion of DACreate1d(),
$             DACreate2d(), and DACreate3d()

   Notes:
   Use DAGetCorners() and DAGetGhostCorners() to get the starting
   and ending grid points (ghost points) in each direction.

.keywords: distributed array, view, visualize

.seealso: ViewerFileOpenASCII(), ViewerDrawOpenX(), DAGetInfo(), DAGetCorners(),
          DAGetGhostCorners()
@*/
int DAView(DA da, Viewer v)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  return (*da->view)((PetscObject)da,v);
}  

/*@
   DAGetInfo - Gets information about a given distributed array.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  dim - dimension of the distributed array (1, 2, or 3)
.  M, N, P - global dimension in each direction of the array
.  m, n, p - corresponding number of procs in each dimension
.  w - number of degrees of freedom per node
.  s - stencil width

   Note:
   Use PETSC_NULL in place of any output parameter that is not of interest.

.keywords: distributed array, get, information

.seealso: DAView()
@*/
int DAGetInfo(DA da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *w,int *s)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (dim != PETSC_NULL) *dim = da->dim;
  if (M != PETSC_NULL)   *M   = da->M;
  if (N != PETSC_NULL)   *N   = da->N;
  if (P != PETSC_NULL)   *P   = da->P;
  if (m != PETSC_NULL)   *m   = da->m;
  if (n != PETSC_NULL)   *n   = da->n;
  if (p != PETSC_NULL)   *p   = da->p;
  if (w != PETSC_NULL)   *w   = da->w;
  if (s != PETSC_NULL)   *s   = da->s;
  return 0;
}  

