/* $Revision: 1.2 $ */
/*
 *
 * char *commname = ALICE_Connect(char *machine,int port) 
 *
 */

#include "mex.h"

/* Input Arguments */

#define	T_IN	prhs[0]
#define	Y_IN	prhs[1]


/* Output Arguments */

#define	YP_OUT	plhs[0]

void mexFunction(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )
{

  
  if (nrhs != 2) {
    mexErrMsgTxt("ALICE_Connect requires two input arguments.");
  } else if (nlhs > 1) {
    mexErrMsgTxt("ALICE_Connect requires one output argument.");
  }
  
  return;
}


