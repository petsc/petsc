/* $Revision: 1.1 $ */
/*
 *
 * char **list = ALICE_Comm_get_memory_list(int comm)
 *
 */

#include "mex.h"
#include "alicemem.h"
#include <string.h>

void mxALICE_Comm_get_memory_list(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )
{
  char       **list,**tlist;
  int        ierr,verbose = 1,comm,nlist;

  if (nrhs != 1) {
    mexErrMsgTxt("ALICE_Comm_get_memory_list requires one input arguments.");
  } else if (nlhs > 1) {
    mexErrMsgTxt("ALICE_Comm_get_memory_list requires one output argument.");
  } 

  comm = *mxGetPr(prhs[0]);

  if (verbose) {
    printf("Comm %d\n",comm);
  }

  ierr = ALICE_Comm_get_memory_list(comm,&list);
  if (ierr) {
    char *err;
    ALICE_Explain_error(ierr,&err);
    printf("%s\n",err);
    mexErrMsgTxt("ALICE_Comm_get_memory_list could not get memory list.");
  }

  if (list) {

    /* count number of strings */
    tlist = list;
    nlist  = 0;
    while (*tlist++) nlist++;

    plhs[0] = mxCreateCharMatrixFromStrings(nlist,(const char **)list);
  }


  return;
}


