/* $Revision: 1.1 $ */
/*
 *
 * char *commname = ALICE_Connect(char *machine,int port) 
 *
 */

#include "mex.h"

void mexFunction(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )
{
  char host[256],**comms,**tcomms;
  int  port,ierr,verbose = 1,ncomm;
  
  if (nrhs != 2 && nrhs != 1) {
    mexErrMsgTxt("ALICE_Connect requires one or two input arguments.");
  } else if (nlhs > 1) {
    mexErrMsgTxt("ALICE_Connect requires one output argument.");
  } else if (!mxIsChar(prhs[0])) {
    mexErrMsgTxt("ALICE_Connect requires first Argument be character.");
  }

  if (nrhs == 1) {
    port = -1;
  } else {
    port = *mxGetPr(prhs[1]);
  }
  mxGetString(prhs[0],host,256);

  if (verbose) {
    printf("Host %s port %d\n",host,port);
  }

  ierr = ALICE_Connect(host,port,&comms);
  if (verbose && comms) {
    printf("Communicators found: \n");
    tcomms = comms;
    while (*tcomms) {
      printf("%s\n",*tcomms);
      tcomms++;
    }
  }

  if (comms) {

    /* count number of strings */
    tcomms = comms;
    ncomm  = 0;
    while (*tcomms++) ncomm++;

    plhs[0] = mxCreateCharMatrixFromStrings(ncomm,(const char **)comms);
  }


  return;
}


