#ifndef lint
static char vcid[] = "$Id: senddense.c,v 1.12 1995/07/17 20:42:41 bsmith Exp bsmith $";
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
  int t = viewer->port,type = DENSEREAL,value;
  if (SOCKWriteInt_Private(t,&type,1)) 
                           SETERRQ(1,"ViewerMatlabPutArray_Private: type");
  if (SOCKWriteInt_Private(t,&m,1)) 
                         SETERRQ(1,"ViewerMatlabPutArray_Private: columns");
  if (SOCKWriteInt_Private(t,&n,1)) 
                             SETERRQ(1,"ViewerMatlabPutArray_Private: rows");
#if !defined(PETSC_COMPLEX)
  value = 0;
  if (SOCKWriteInt_Private(t,&value,1))
                          SETERRQ(1,"ViewerMatlabPutArray_Private: complex");
  if (SOCKWriteDouble_Private(t,array,m*n))
                          SETERRQ(1,"ViewerMatlabPutArray_Private:array");
#else
  value = 1;
  if (SOCKWriteInt_Private(t,&value,1)) 
                          SETERRQ(1,"ViewerMatlabPutArray_Private:complex");
  if (SOCKWriteDouble_Private(t,(double*)array,2*m*n)) 
                             SETERRQ(1,"ViewerMatlabPutArray_Private:array");
#endif
  return 0;
}

