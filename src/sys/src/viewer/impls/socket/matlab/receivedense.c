#ifndef lint
static char vcid[] = "$Id: receivedense.c,v 1.3 1995/03/06 04:39:21 bsmith Exp bsmith $";
#endif
/* 
   This is part of the MatlabSockettool Package. It is called by 
 the receive.mex4 Matlab program. 
  
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92

  Since this is called from Matlab it cannot be compiled with C++.
*/

#include <stdio.h>
#include "sys.h"
#include "mex.h"
#define ERROR(a) {fprintf(stderr,"RECEIVE %s \n",a); return -1;}
/*-----------------------------------------------------------------*/
int ReceiveDenseMatrix(Matrix *plhs[],int t)
{
  int    m,n,compx = 0,i;
  
  /* get size of matrix */
  if (PetscBinaryRead(t,&m,1,BINARY_INT))   ERROR("reading number columns"); 
  if (PetscBinaryRead(t,&n,1,BINARY_INT))   ERROR("reading number rows"); 
  if (PetscBinaryRead(t,&compx,1,BINARY_INT))   ERROR("reading number rows"); 
  /*allocate matrix */
  plhs[0]  = mxCreateFull(m, n, compx);
  /* read in matrix */
  if (!compx) {
    if (PetscBinaryRead(t,mxGetPr(plhs[0]),n*m,BINARY_DOUBLE)) ERROR("read dense matrix");
  }
  else {
    for ( i=0; i<n*m; i++ ) {
      if (PetscBinaryRead(t,mxGetPr(plhs[0])+i,1,BINARY_DOUBLE))ERROR("read dense matrix");
      if (PetscBinaryRead(t,mxGetPi(plhs[0])+i,1,BINARY_DOUBLE))ERROR("read dense matrix");
    }
  }
  return 0;
}

    
 
