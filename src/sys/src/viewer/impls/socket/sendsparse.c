/* This is part of the MatlabSockettool package. Here are the routines
   to send a sparse matrix to Matlab.

   The sparse matrix is stored in Column oriented form with the 
   diagonal stored separately. The matrix need not be symmetric 
   of triangular.

    Usage: Fortran: putspa(machine, portnumber, m, n, nz, D, A, c, r)
           C:       putsparse(machine, portnumber, m, n, nz, D, A, c, r)

       char   *machine    e.g. "condor"
       int    portnumber  [  5000 < portnumber < 5010 ]
       int    m,n         number of rows and columns in matrix
       int    nz          number of off diagonal nonzeros
       double D[m]        the diagonal elements of the matrix
       double A[nz]       the off diagonal elements of the matrix.
       int    c[m+1]      pointers to the beginning of the i-th 
                          column in A and r.
       int    r[nz]       for each off diagonal element in A this 
                          points to the row it is in.

        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/
#include <stdio.h>
#include "matlab.h"


/*--------------------------------------------------------------*/
#define ERROR(a) {fprintf(stderr,"SEND: %s \n",a); return -1;}
putsparse(Viewer viewer,int m,int n,int nnz,double *d,
          double *v,int *c,int *r)
{
  int t = viewer->port,type = SPARSEREAL;
   
  /* write the type of matrix */  
  if (write_int(t,&type,1))       ERROR("writing type");
  /* write matrix */
  if (write_int(t,&m,1))          ERROR("writing number rows");
  if (write_int(t,&n,1))          ERROR("writing number columns");
  if (write_int(t,&nnz,1))        ERROR("writing number nonzeros");
  if (write_double(t,v,nnz))      ERROR("writing elements");
  if (write_int(t,c,m+1))         ERROR("writing column pointers");
  if (write_int(t,r,nnz))         ERROR("writing row pointers");
  return 0;
}

