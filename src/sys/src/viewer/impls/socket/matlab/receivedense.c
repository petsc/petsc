#ifndef lint
static char vcid[] = "$Id: $";
#endif
/* 
   This is part of the MatlabSockettool Package. It is called by 
 the receive.mex4 Matlab program. 
  
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/

#include <stdio.h>
#include "mex.h"
#define ERROR(a) {fprintf(stderr,"RECEIVE %s \n",a); return -1;}
/*-----------------------------------------------------------------*/
int ReceiveDenseMatrix(Matrix *plhs[],int t)
{
  int    m,n;

  /* get size of matrix */
  if (read_int(t,&m,1))   ERROR("reading number columns"); 
  if (read_int(t,&n,1))   ERROR("reading number rows"); 
  /*allocate matrix */
  plhs[0]  = mxCreateFull(m, n, 0);
  /* read in matrix */
  if (read_double(t,mxGetPr(plhs[0]),n*m)) ERROR("read dense matrix");
  return 0;
}

    
 
