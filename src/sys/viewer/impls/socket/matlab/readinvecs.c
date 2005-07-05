
/*    Reads in PETSc vectors from a PETSc binary file into matlab

  Since this is called from Matlab it cannot be compiled with C++.
  Modified Sept 28, 2003 RFK: updated obsolete mx functions.
*/


#include "petscsys.h"
#include "petscvec.h"
#include "mex.h"
#include <fcntl.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (PETSC_HAVE_IO_H)
#include <io.h>
#endif
#if defined(PETSC_HAVE_STRINGS_H)
#include <strings.h>
#endif
#if defined(PETSC_HAVE_STRING_H)
#include <string.h>
#endif
#if defined(PETSC_HAVE_STROPTS_H)
#include <stropts.h>
#endif

#define PETSC_MEX_ERROR(a) {fprintf(stdout,"ReadInVecs %s \n",a); return -1;}
/*-----------------------------------------------------------------*/
/*
       Reads in a single vector
*/
#undef __FUNCT__  
#define __FUNCT__ "ReadInVecs"
PetscErrorCode ReadInVecs(mxArray *plhs[],int t,int dim,int *dims)
{
  int          cookie = 0,M,i;
  mxComplexity compx = mxREAL;

  /* get size of matrix */
  if (PetscBinaryRead(t,&cookie,1,PETSC_INT))   return -1;  /* finished reading file */
  if (cookie != VEC_FILE_COOKIE) PETSC_MEX_ERROR("could not read vector cookie");
  if (PetscBinaryRead(t,&M,1,PETSC_INT))        PETSC_MEX_ERROR("reading number rows"); 
  
  if (dim == 1) {
    plhs[0]  = mxCreateDoubleMatrix(M,1,mxREAL);
  } else if (dim == 2) {
    if (dims[0]*dims[1] != M) {
      printf("PETSC_MEX_ERROR: m %d * n %d != M %d\n",dims[0],dims[1],M);
      return -1;
    }
    plhs[0]  = mxCreateDoubleMatrix(dims[0],dims[1],mxREAL);
  } else {
    plhs[0] = mxCreateNumericArray(dim,dims,mxDOUBLE_CLASS,mxREAL);
  }

  /* read in matrix */
  if (compx == mxREAL) { /* real */
    if (PetscBinaryRead(t,mxGetPr(plhs[0]),M,PETSC_DOUBLE)) PETSC_MEX_ERROR("read dense matrix");
  } else { /* complex, currently not used */
    for (i=0; i<M; i++) {
      if (PetscBinaryRead(t,mxGetPr(plhs[0])+i,1,PETSC_DOUBLE)) PETSC_MEX_ERROR("read dense matrix");
      if (PetscBinaryRead(t,mxGetPi(plhs[0])+i,1,PETSC_DOUBLE)) PETSC_MEX_ERROR("read dense matrix");
    }
  }
  return 0;
}

#undef PETSC_MEX_ERROR
#define PETSC_MEX_ERROR(a) {fprintf(stdout,"ReadInVecs %s \n",a); return;}
/*-----------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "mexFunction"
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  static int fd = -1,dims[4],dim = 1,dof;
  char       filename[1024],buffer[1024];
  int        err,d2,d3,d4;
  FILE       *file;

  /* check output parameters */
  if (nlhs != 1) PETSC_MEX_ERROR("Receive requires one output argument.");
  if (fd == -1) {
    if (!mxIsChar(prhs[0])) PETSC_MEX_ERROR("First arg must be string.");
  
    /* open the file */
    mxGetString(prhs[0],filename,256);
    fd = open(filename,O_RDONLY,0);

    strcat(filename,".info");
    file = fopen(filename,"r");
    if (file) {
      fgets(buffer,1024,file);
      if (!strncmp(buffer,"-daload_info",12)) {
        sscanf(buffer,"-daload_info %d,%d,%d,%d,%d,%d,%d,%d\n",&dim,&dims[0],&dims[1],&dims[2],&dof,&d2,&d3,&d4);
        if (dof > 1) {
          dim++;
          dims[3] = dims[2];
          dims[2] = dims[1];
          dims[1] = dims[0];
          dims[0] = dof;
        }
      }
      fclose(file);
    }
  }

  /* read in the next vector */
  err = ReadInVecs(plhs,fd,dim,dims);

  if (err) {  /* file is finished so close and allow a restart */
    close(fd);
    fd = -1; 
  }
  return;
}


 
