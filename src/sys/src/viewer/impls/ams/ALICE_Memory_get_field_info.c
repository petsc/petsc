/* $Revision: 1.1 $ */
/*
 *
 * char **list = ALICE_Memory_get_field_list(int memory)
 *
 */

#include "mex.h"
#include "alicemem.h"
#include <string.h>

void mxALICE_Memory_get_field_list(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )
{
  char       **list,**tlist;
  int        ierr,verbose = 1,memory,nlist;

  if (nrhs != 1) {
    mexErrMsgTxt("ALICE_Memory_get_field_list requires one input arguments.");
  } else if (nlhs != 1) {
    mexErrMsgTxt("ALICE_Memory_get_field_list requires one output argument.");
  } 

  memory = *mxGetPr(prhs[0]);

  if (verbose) {
    printf("Memory %d\n",memory);
  }

  ierr = ALICE_Memory_get_field_list(memory,&list);
  if (ierr) {
    char *err;
    ALICE_Explain_error(ierr,&err);
    printf("%s\n",err);
    mexErrMsgTxt("ALICE_Memory_get_field_list could not get field list.");
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


