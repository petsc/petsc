/* $Revision: 1.1 $ */
/*
 *
 * ALICE_Comm_detach(int comm) 
 *
 */

#include "mex.h"
#include "alicemem.h"
#include <string.h>

void mxALICE_Comm_detach(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )
{
  int        ierr,verbose = 1;
  ALICE_Comm comm;

  if (nrhs != 1) {
    mexErrMsgTxt("ALICE_Comm_detach requires one input arguments.");
  } else if (nlhs > 0) {
    mexErrMsgTxt("ALICE_Comm_detach requires zero output argument.");
  } 

  comm = *(mxGetPr(prhs[0]));

  ierr = ALICE_Comm_detach(comm);
  if (ierr) {
    char *err;
    ALICE_Explain_error(ierr,&err);
    printf("%s\n",err);
    mexErrMsgTxt("ALICE_Comm_detach could not detach .");
  }

  return;
}


