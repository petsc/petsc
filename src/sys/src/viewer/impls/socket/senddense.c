#ifndef lint
static char vcid[] = "$Id: senddense.c,v 1.15 1996/03/19 21:28:40 bsmith Exp bsmith $";
#endif
/*
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/
#include "src/viewer/impls/matlab/matlab.h"

/*
   ViewerMatlabPutArray_Private - Passes an array to a Matlab viewer.

  Input Paramters:
.  viewer - obtained from ViewerMatlabOpen()
.  m, n - number of rows and columns of array
.  array - the array stored in Fortran 77 style (matrix or vector data) 

   Notes:
   Most users should not call this routine, but instead should employ
   either
$     MatView(Mat matrix,Viewer viewer)
$
$              or
$
$     VecView(Vec vector,Viewer viewer)

.keywords: Viewer, Matlab, put, dense, array, vector

.seealso: ViewerMatlabOpen(), MatView(), VecView()
*/
int ViewerMatlabPutArray_Private(Viewer viewer,int m,int n,Scalar *array)
{
  int ierr,t = viewer->port,type = DENSEREAL,value;
  ierr = PetscBinaryWrite(t,&type,1,BINARY_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,BINARY_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,BINARY_INT,0); CHKERRQ(ierr); 
#if !defined(PETSC_COMPLEX)
  value = 0;
#else
  value = 1;
#endif
  ierr = PetscBinaryWrite(t,&value,1,BINARY_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m*n,BINARY_SCALAR,0);
  return 0;
}

