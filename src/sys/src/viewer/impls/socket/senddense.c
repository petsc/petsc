#ifndef lint
static char vcid[] = "$Id: senddense.c,v 1.11 1995/06/08 03:11:27 bsmith Exp bsmith $";
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
  if (write_int(t,&type,1)) SETERRQ(1,"ViewerMatlabPutArray_Private: type");
  if (write_int(t,&m,1))    SETERRQ(1,"ViewerMatlabPutArray_Private: columns");
  if (write_int(t,&n,1))    SETERRQ(1,"ViewerMatlabPutArray_Private: rows");
#if !defined(PETSC_COMPLEX)
  value = 0;
  if (write_int(t,&value,1))SETERRQ(1,"ViewerMatlabPutArray_Private: complex");
  if (write_double(t,array,m*n))SETERRQ(1,"ViewerMatlabPutArray_Private:array");
#else
  value = 1;
  if (write_int(t,&value,1)) SETERRQ(1,"ViewerMatlabPutArray_Private:complex");
  if (write_double(t,(double*)array,2*m*n)) 
                             SETERRQ(1,"ViewerMatlabPutArray_Private:array");
#endif
  return 0;
}

