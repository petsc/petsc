#ifndef lint
static char vcid[] = "$Id: senddense.c,v 1.13 1995/07/20 04:00:08 bsmith Exp bsmith $";
#endif
/*
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/
#include "matlab.h"

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
  ierr = SYWrite(t,&type,1,SYINT,0); CHKERRQ(ierr);
  ierr = SYWrite(t,&m,1,SYINT,0); CHKERRQ(ierr);
  ierr = SYWrite(t,&n,1,SYINT,0); CHKERRQ(ierr); 
#if !defined(PETSC_COMPLEX)
  value = 0;
#else
  value = 1;
#endif
  ierr = SYWrite(t,&value,1,SYINT,0); CHKERRQ(ierr);
  ierr = SYWrite(t,array,m*n,SYSCALAR,0);
  return 0;
}

