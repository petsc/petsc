/*$Id: daview.c,v 1.39 2000/01/11 21:03:19 bsmith Exp bsmith $*/
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DAView"
/*@C
   DAView - Visualizes a distributed array object.

   Collective on DA

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

   Level: beginner

   Notes:
   Use DAGetCorners() and DAGetGhostCorners() to get the starting
   and ending grid points (ghost points) in each direction.

.keywords: distributed array, view, visualize

.seealso: ViewerASCIIOpen(), ViewerDrawOpen(), DAGetInfo(), DAGetCorners(),
          DAGetGhostCorners()
@*/
int DAView(DA da,Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (!viewer) viewer = VIEWER_STDOUT_WORLD;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = (*da->view)(da,viewer);CHKERRQ(ierr);
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

   Level: beginner
  
   Note:
   Use PETSC_NULL in place of any output parameter that is not of interest.

.keywords: distributed array, get, information

.seealso: DAView()
@*/
int DAGetInfo(DA da,int *dim,int *M,int *N,int *P,int *m,int *n,int *p,int *dof,int *s,DAPeriodicType *wrap,DAStencilType *st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (dim)  *dim  = da->dim;
  if (M)    *M    = da->M;
  if (N)    *N    = da->N;
  if (P)    *P    = da->P;
  if (m)    *m    = da->m;
  if (n)    *n    = da->n;
  if (p)    *p    = da->p;
  if (dof)  *dof  = da->w;
  if (s)    *s    = da->s;
  if (wrap) *wrap = da->wrap;
  if (st)   *st   = da->stencil_type;
  PetscFunctionReturn(0);
}  

#undef __FUNC__  
#define __FUNC__ "DAView_Binary"
int DAView_Binary(DA da,Viewer viewer)
{
  int            rank,ierr;
  int            i,j,len,dim,m,n,p,dof,swidth,M,N,P;
  DAStencilType  stencil;
  DAPeriodicType periodic;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = DAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&dof,&swidth,&periodic,&stencil);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    FILE *file;

    ierr = ViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
    if (file) {
      char           fieldname[256];

      fprintf(file,"-daload_info %d,%d,%d,%d,%d,%d,%d,%d\n",dim,m,n,p,dof,swidth,stencil,periodic);
      for (i=0; i<dof; i++) {
        if (da->fieldname[i]) {
          ierr = PetscStrncpy(fieldname,da->fieldname[i],256);CHKERRQ(ierr);
          ierr = PetscStrlen(fieldname,&len);CHKERRQ(ierr);
          len  = PetscMin(256,len);CHKERRQ(ierr);
          for (j=0; j<len; j++) {
            if (fieldname[j] == ' ') fieldname[j] = '_';
          }
          fprintf(file,"-daload_fieldname_%d %s\n",i,fieldname);
        }
      }
      if (da->coordinates) { /* save the DA's coordinates */
        fprintf(file,"-daload_coordinates\n");
      }
    }
  } 

  /* save the coordinates if they exist to disk (in the natural ordering) */
  if (da->coordinates) {
    DA  dac;
    int *lx,*ly,*lz;
    Vec natural;

    /* create the appropriate DA to map to natural ordering */
    ierr = DAGetOwnershipRange(da,&lx,&ly,&lz);CHKERRQ(ierr);
    if (dim == 1) {
      ierr = DACreate1d(comm,DA_NONPERIODIC,m,dim,0,lx,&dac);CHKERRQ(ierr); 
    } else if (dim == 2) {
      ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,m,n,M,N,dim,0,lx,ly,&dac);CHKERRQ(ierr); 
    } else if (dim == 3) {
      ierr = DACreate3d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,m,n,p,M,N,P,dim,0,lx,ly,lz,&dac);CHKERRQ(ierr); 
    } else {
      SETERRQ1(1,1,"Dimension is not 1 2 or 3: %d\n",dim);
    }
    ierr = DACreateNaturalVector(dac,&natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(dac,da->coordinates,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(dac,da->coordinates,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = VecView(natural,viewer);CHKERRQ(ierr);
    ierr = VecDestroy(natural);CHKERRQ(ierr);
    ierr = DADestroy(dac);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}





