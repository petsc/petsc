#ifndef lint
static char vcid[] = "$Id: $";
#endif
/* This is part of the MatlabSockettool package. Here are the routines
   to send a dense matrix to Matlab.

 
    Usage: Fortran: putmat(machine, portnumber, m, n, matrix)
           C:       putmatrix(machine, portnumber, m, n, matrix)

       char   *machine    e.g. "condor"
       int    portnumber  [  5000 < portnumber < 5010 ]
       int    m,n         number of rows and columns in matrix
       double *matrix     fortran style matrix
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/
#include "matlab.h"

/*@
    ViewerMatlabPutArray - Passes an array to a Matlab viewer.
         This is not usually used by an application programmer,
         instead, he or she would call either VecView() or MatView().

  Input Paramters:
.  viewer - obtained from ViewerMatlabOpen()
.  m, n - number of rows and columns of array
.  matrix - the array stored in Fortran 77 style.

@*/
int ViewerMatlabPutArray(Viewer viewer,int m,int n,double *matrix)
{
  int t = viewer->port,type = DENSEREAL;
  if (write_int(t,&type,1))       SETERR(1,"writing type");
  if (write_int(t,&m,1))          SETERR(1,"writing number columns");
  if (write_int(t,&n,1))          SETERR(1,"writing number rows");
  if (write_double(t,matrix,m*n)) SETERR(1,"writing dense matrix");
  return 0;
}

