#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: daview.c,v 1.29 1999/03/07 17:30:00 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DAView"
/*@C
   DAView - Visualizes a distributed array object.

   Collective on DA, unless Viewer is VIEWER_STDOUT_SELF

   Input Parameters:
+  da - the distributed array
-  ptr - an optional visualization context

   Notes:
   The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
.     VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 
-     VIEWER_DRAW_WORLD - to default window

   The user can open alternative visualization contexts with
+    ViewerASCIIOpen() - Outputs vector to a specified file
-    ViewerDrawOpen() - Outputs vector to an X window display

   Default Output Format:
  (for 3d arrays)
.vb
   Processor [proc] M  N  P  m  n  p  w  s
   X range: xs xe, Y range: ys, ye, Z range: zs, ze

   where
      M,N,P - global dimension in each direction of the array
      m,n,p - corresponding number of procs in each dimension 
      w - number of degrees of freedom per node
      s - stencil width
      xs, xe - internal local starting/ending grid points
               in x-direction, (augmented to handle multiple 
               degrees of freedom per node)
      ys, ye - local starting/ending grid points in y-direction
      zs, ze - local starting/ending grid points in z-direction
.ve

   Options Database Key:
.  -da_view - Calls DAView() at the conclusion of DACreate1d(),
              DACreate2d(), and DACreate3d()

   Notes:
   Use DAGetCorners() and DAGetGhostCorners() to get the starting
   and ending grid points (ghost points) in each direction.

.keywords: distributed array, view, visualize

.seealso: ViewerASCIIOpen(), ViewerDrawOpen(), DAGetInfo(), DAGetCorners(),
          DAGetGhostCorners()
@*/
int DAView(DA da, Viewer v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = (*da->view)(da,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

#undef __FUNC__  
#define __FUNC__ "DAGetInfo"
/*@C
   DAGetInfo - Gets information about a given distributed array.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  dim     - dimension of the distributed array (1, 2, or 3)
.  M, N, P - global dimension in each direction of the array
.  m, n, p - corresponding number of procs in each dimension
.  dof     - number of degrees of freedom per node
.  s       - stencil width
.  wrap    - type of periodicity, on of DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, 
             DA_XYPERIODIC, DA_XYZPERIODIC, DA_XZPERIODIC, DA_YZPERIODIC,DA_ZPERIODIC
-  st      - stencil type, either DA_STENCIL_STAR or DA_STENCIL_BOX
  
   Note:
   Use PETSC_NULL in place of any output parameter that is not of interest.

.keywords: distributed array, get, information

.seealso: DAView()
@*/
int DAGetInfo(DA da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *dof,int *s,DAPeriodicType *wrap,
              DAStencilType *st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (dim != PETSC_NULL)  *dim  = da->dim;
  if (M != PETSC_NULL)    *M    = da->M;
  if (N != PETSC_NULL)    *N    = da->N;
  if (P != PETSC_NULL)    *P    = da->P;
  if (m != PETSC_NULL)    *m    = da->m;
  if (n != PETSC_NULL)    *n    = da->n;
  if (p != PETSC_NULL)    *p    = da->p;
  if (dof != PETSC_NULL)  *dof  = da->w;
  if (s != PETSC_NULL)    *s    = da->s;
  if (wrap != PETSC_NULL) *wrap = da->wrap;
  if (st != PETSC_NULL)   *st   = da->stencil_type;
  PetscFunctionReturn(0);
}  

#undef __FUNC__  
#define __FUNC__ "DAView_Binary"
int DAView_Binary(DA da,Viewer viewer)
{
  int            rank,ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    FILE *file;

    ierr = ViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
    if (file) {
      int            dim,m,n,p,dof,swidth;
      DAStencilType  stencil;
      DAPeriodicType periodic;

      ierr = DAGetInfo(da,&dim,&m,&n,&p,0,0,0,&dof,&swidth,&periodic,&stencil);CHKERRQ(ierr);
      fprintf(file,"-daload_info %d,%d,%d,%d,%d,%d,%d,%d\n",dim,m,n,p,dof,swidth,stencil,periodic);
    }
  } 
  PetscFunctionReturn(0);
}





