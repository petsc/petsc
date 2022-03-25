
static char help[] = "Writes an array to a file, then reads an array from a file, then forms a vector.\n\n";

/*
    This uses the low level PetscBinaryWrite() and PetscBinaryRead() to access a binary file. It will not work in parallel!

    We HIGHLY recommend using instead VecView() and VecLoad() to read and write Vectors in binary format (which also work in parallel). Then you can use
    share/petsc/matlab/PetscBinaryRead() and share/petsc/matlab/PetscBinaryWrite() to read (or write) the vector into MATLAB.

    Note this also works for matrices with MatView() and MatLoad().
*/
#include <petscvec.h>

int main(int argc,char **args)
{
  PetscMPIInt    size;
  int            fd;
  PetscInt       i,m = 10,sz;
  PetscScalar    *avec,*array;
  Vec            vec;
  PetscViewer    view_out,view_in;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  /* ---------------------------------------------------------------------- */
  /*          PART 1: Write some data to a file in binary format            */
  /* ---------------------------------------------------------------------- */

  /* Allocate array and set values */
  PetscCall(PetscMalloc1(m,&array));
  for (i=0; i<m; i++) array[i] = i*10.0;

  /* Open viewer for binary output */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_WRITE,&view_out));
  PetscCall(PetscViewerBinaryGetDescriptor(view_out,&fd));

  /* Write binary output */
  PetscCall(PetscBinaryWrite(fd,&m,1,PETSC_INT));
  PetscCall(PetscBinaryWrite(fd,array,m,PETSC_SCALAR));

  /* Destroy the output viewer and work array */
  PetscCall(PetscViewerDestroy(&view_out));
  PetscCall(PetscFree(array));

  /* ---------------------------------------------------------------------- */
  /*          PART 2: Read data from file and form a vector                 */
  /* ---------------------------------------------------------------------- */

  /* Open input binary viewer */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_READ,&view_in));
  PetscCall(PetscViewerBinaryGetDescriptor(view_in,&fd));

  /* Create vector and get pointer to data space */
  PetscCall(VecCreate(PETSC_COMM_SELF,&vec));
  PetscCall(VecSetSizes(vec,PETSC_DECIDE,m));
  PetscCall(VecSetFromOptions(vec));
  PetscCall(VecGetArray(vec,&avec));

  /* Read data into vector */
  PetscCall(PetscBinaryRead(fd,&sz,1,NULL,PETSC_INT));
  PetscCheckFalse(sz <=0,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Error: Must have array length > 0");

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"reading data in binary from input.dat, sz =%" PetscInt_FMT " ...\n",sz));
  PetscCall(PetscBinaryRead(fd,avec,sz,NULL,PETSC_SCALAR));

  /* View vector */
  PetscCall(VecRestoreArray(vec,&avec));
  PetscCall(VecView(vec,PETSC_VIEWER_STDOUT_SELF));

  /* Free data structures */
  PetscCall(VecDestroy(&vec));
  PetscCall(PetscViewerDestroy(&view_in));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:

TEST*/
