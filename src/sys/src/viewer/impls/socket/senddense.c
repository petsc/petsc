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
#include <stdio.h>
#include "matlab.h"


/*--------------------------------------------------------------*/
#define ERROR(a) {fprintf(stderr,"SEND: %s \n",a); return -1;}
putmatrix(Viewer viewer,int m,int n,double *matrix)
{
  int t = viewer->port,type = DENSEREAL;

  /* write the type of matrix */  
  if (write_int(t,&type,1))       ERROR("writing type");
  /* write matrix */
  if (write_int(t,&m,1))          ERROR("writing number columns");
  if (write_int(t,&n,1))          ERROR("writing number rows");
  if (write_double(t,matrix,m*n)) ERROR("writing dense matrix");
  return 0;
}
