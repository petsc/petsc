/*
 
  This is a MATLAB Mex program which waits at a particular 
  portnumber until a matrix arrives, it then returns to 
  matlab with that matrix.

  Usage: A = receive(portnumber);  portnumber obtained with openport();
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/

#include <stdio.h>
#include "mex.h"
#include "matlab.h"
#define ERROR(a) {fprintf(stderr,"RECEIVE: %s \n",a); return ;}
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
void mexFunction(int nlhs, Matrix *plhs[], int nrhs, Matrix *prhs[])
{
  int    type,t;

  /* check output parameters */
  if (nlhs != 1) ERROR("Receive requires one output argument.");

  if (nrhs == 0) ERROR("Receive requires one input argument.");
  t = (int) *mxGetPr(prhs[0]);

  /* get type of matrix */
  if (read_int(t,&type,1))   ERROR("reading type"); 

  if (type == DENSEREAL) ReceiveDenseMatrix(plhs,t);
  if (type == DENSECHARACTER) {
    if (ReceiveDenseMatrix(plhs,t)) return;
    mxSetDispMode(plhs[0],1);
  }
  if (type == SPARSEREAL) ReceiveSparseMatrix(plhs,t); 
  return;
}
/*-----------------------------------------------------------------*/
int read_int(int t,char *buff,int n)
{
  return(read_data(t,buff,n*sizeof(int)));
}
/*-----------------------------------------------------------------*/
int read_double(int t,char *buff,int n)
{
  return(read_data(t,buff,n*sizeof(double)));
}
/*-----------------------------------------------------------------*/
int read_data(int t,char *buff,int n)
{
   int bcount,br;

   bcount = 0;
   br = 0;
   while ( bcount < n ) {
     if ( (br=read(t,buff,n-bcount)) > 0 ) {
        bcount += br;
        buff += br;
     }
     else {
       perror("RECEIVE: error reading");
       return(-1);
     }
   }
   return(0);
}
