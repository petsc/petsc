/*

    This is the equivalent of MATLAB's fread() only on sockets instead of
   binary files.
*/

#include <petscsys.h>
#include <../src/sys/classes/viewer/impls/socket/socket.h>
#include <mex.h>

PetscErrorCode PetscBinaryRead(int, void *p, int, int *, PetscDataType);

#define PETSC_MEX_ERROR(a) \
  { \
    fprintf(stdout, "sread: %s \n", a); \
    return; \
  }
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
PETSC_EXTERN void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int            fd, cnt, dt;
  PetscErrorCode ierr;

  /* check output parameters */
  if (nlhs != 1) PETSC_MEX_ERROR("Receive requires one output argument.");
  if (nrhs != 3) PETSC_MEX_ERROR("Receive requires three input arguments.");
  fd  = (int)mxGetScalar(prhs[0]);
  cnt = (int)mxGetScalar(prhs[1]);
  dt  = (PetscDataType)mxGetScalar(prhs[2]);

  if (dt == PETSC_DOUBLE) {
    plhs[0] = mxCreateDoubleMatrix(1, cnt, mxREAL);
    ierr    = PetscBinaryRead(fd, mxGetPr(plhs[0]), cnt, NULL, (PetscDataType)dt);
    if (ierr) PETSC_MEX_ERROR("Unable to receive double items.");
  } else if (dt == PETSC_INT) {
    plhs[0] = mxCreateNumericMatrix(1, cnt, mxINT32_CLASS, mxREAL);
    ierr    = PetscBinaryRead(fd, mxGetPr(plhs[0]), cnt, NULL, (PetscDataType)dt);
    if (ierr) PETSC_MEX_ERROR("Unable to receive int items.");
  } else if (dt == PETSC_CHAR) {
    char *tmp = (char *)mxMalloc(cnt * sizeof(char));
    ierr      = PetscBinaryRead(fd, tmp, cnt, NULL, (PetscDataType)dt);
    if (ierr) PETSC_MEX_ERROR("Unable to receive char items.");
    plhs[0] = mxCreateStringFromNChars(tmp, cnt);
    mxFree(tmp);
  } else PETSC_MEX_ERROR("Unknown datatype.");
  return;
}

int main(int argc, char **argv)
{
  return 0;
}
