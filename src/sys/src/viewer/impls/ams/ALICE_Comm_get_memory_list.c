/* $Revision: 1.1 $ */
/*
 *
 * int comm = ALICE_Comm_attach(char *name) 
 *
 */

#include "mex.h"
#include "alicemem.h"
#include <string.h>

void mexFunction(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )
{
  char       name[256],*lname;
  int        ierr,verbose = 1,ncomm,one = 1;
  ALICE_Comm comm;

  if (nrhs != 1) {
    mexErrMsgTxt("ALICE_Comm_attach requires one input arguments.");
  } else if (nlhs > 1) {
    mexErrMsgTxt("ALICE_Comm_attach requires one output argument.");
  } else if (!mxIsChar(prhs[0])) {
    mexErrMsgTxt("ALICE_Comm_attach requires argument be character.");
  }

  mxGetString(prhs[0],name,256);


  lname = strtok(name,"|");
  if (verbose) {
    printf("Comm Name %s \n",lname);
  }

  ierr = ALICE_Comm_attach(lname,&comm);
  if (ierr) {
    char *err;
    ALICE_Explain_error(ierr,&err);
    printf("%s\n",err);
    mexErrMsgTxt("ALICE_Comm_attach could not attach .");
  }
  plhs[0] = mxCreateDoubleMatrix(1,1,0);
  *(mxGetPr(plhs[0])) = (double) comm;


  return;
}


