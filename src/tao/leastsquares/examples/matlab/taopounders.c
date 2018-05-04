#include <stdio.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  char *cbfun;
  double *xp, *op;
  int n, m, len;
  int i;

  /* 
     The number of input arguments (nrhs) should be 4
     0 - dimension of the inputs problem (n)
     1 - starting point of length n
     2 - dimension of the residuals (m)
     3 - character string containing name of the function to evaluate the residuals

     The number of output arguments (nlhs) should be 1
     0 - solution (n)
  */

  if (nrhs < 4) {
    mexErrMsgTxt ("Not enough input arguments.");
  }

  if (nlhs < 1) {
    mexErrMsgTxt ("Not enough output arguments.");
  }

  n = (int) mxGetScalar(prhs[0]);
  xp = mxGetPr(prhs[1]);

  m = (int) mxGetScalar(prhs[2]);

  if (!mxIsClass(prhs[3], "function_handle")) {
    mexPrintf("function handle\n");
  }

  plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
  op = mxGetPr(plhs[0]);
  
  for (i = 0; i < n; ++i) {
    op[i] = xp[i];
    mexPrintf("%5.4e\n", op[i]);
  }

  return;
}

