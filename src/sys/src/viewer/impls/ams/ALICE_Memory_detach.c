/* $Revision: 1.1 $ */
/*
 *
 * ALICE_Memory memory = ALICE_Comm_attach(ALICE_Comm, char *name) 
 *
 */

#include "mex.h"
#include "alicemem.h"
#include <string.h>

void mxALICE_Memory_attach(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )
{
  char         name[256],*lname;
  int          ierr,verbose = 1,ncomm,one = 1;
  ALICE_Comm   comm;
  ALICE_Memory memory;

  if (nrhs != 2) {
    mexErrMsgTxt("ALICE_Memory_attach requires two input arguments.");
  } else if (nlhs != 1) {
    mexErrMsgTxt("ALICE_Memory_attach requires one output argument.");
  } else if (!mxIsChar(prhs[1])) {
    mexErrMsgTxt("ALICE_Memory_attach requires second argument be character.");
  }
  comm = *(mxGetPr(prhs[0]));
  mxGetString(prhs[1],name,256);


  lname = strtok(name,"|");
  if (verbose) {
    printf("Memory Name %s \n",lname);
  }

  ierr = ALICE_Memory_attach(comm,lname,&memory);
  if (ierr) {
    char *err;
    ALICE_Explain_error(ierr,&err);
    printf("%s\n",err);
    mexErrMsgTxt("ALICE_Memory_attach could not attach .");
  }
  plhs[0] = mxCreateDoubleMatrix(1,1,0);
  *(mxGetPr(plhs[0])) = (double) memory;


  return;
}


