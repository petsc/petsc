/* 
   This is part of the MatlabSockettool Package. It is called by 
 the receive.mex4 Matlab program. 
  
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/

#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#include <stropts.h>
#include <math.h>
#include "mex.h"
static int flag;
#define ERROR(a) {fprintf(stderr,"RECEIVE %s \n",a); return 0;}
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
}

    
 
