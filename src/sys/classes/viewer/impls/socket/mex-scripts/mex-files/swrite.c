/*

    This is the equivalent of MATLAB's fwrite() only on sockets instead of
   binary files.
*/

#include <petscsys.h>
#include <../src/sys/classes/viewer/impls/socket/socket.h>
#include <mex.h>

PetscErrorCode PetscBinaryWrite(int, const void *p, int, PetscDataType);

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
  int            i, fd, cnt, dt;
  PetscErrorCode ierr;

  /* check output parameters */
  if (nrhs != 3) PETSC_MEX_ERROR("Receive requires three input arguments.");
  fd  = (int)mxGetScalar(prhs[0]);
  cnt = mxGetNumberOfElements(prhs[1]);
  dt  = (PetscDataType)mxGetScalar(prhs[2]);

  if (dt == PETSC_DOUBLE) {
    ierr = PetscBinaryWrite(fd, mxGetPr(prhs[1]), cnt, (PetscDataType)dt);
    if (ierr) PETSC_MEX_ERROR("Unable to send double items.");
  } else if (dt == PETSC_INT) {
    int    *tmp = (int *)mxMalloc((cnt + 5) * sizeof(int));
    double *t   = mxGetPr(prhs[1]);
    for (i = 0; i < cnt; i++) tmp[i] = (int)t[i];
    ierr = PetscBinaryWrite(fd, tmp, cnt, (PetscDataType)dt);
    if (ierr) PETSC_MEX_ERROR("Unable to send int items.");
    mxFree(tmp);
  } else if (dt == PETSC_CHAR) {
    char *tmp = (char *)mxMalloc((cnt + 5) * sizeof(char));
    mxGetNChars(prhs[1], tmp, cnt + 1);
    ierr = PetscBinaryWrite(fd, tmp, cnt, (PetscDataType)dt);
    if (ierr) PETSC_MEX_ERROR("Unable to send char items.");
    mxFree(tmp);
  } else PETSC_MEX_ERROR("Unknown datatype.");
  return;
}

int main(int argc, char **argv)
{
  return 0;
}
