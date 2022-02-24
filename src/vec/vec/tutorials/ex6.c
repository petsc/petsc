
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
  PetscErrorCode ierr;
  PetscMPIInt    size;
  int            fd;
  PetscInt       i,m = 10,sz;
  PetscScalar    *avec,*array;
  Vec            vec;
  PetscViewer    view_out,view_in;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  /* ---------------------------------------------------------------------- */
  /*          PART 1: Write some data to a file in binary format            */
  /* ---------------------------------------------------------------------- */

  /* Allocate array and set values */
  CHKERRQ(PetscMalloc1(m,&array));
  for (i=0; i<m; i++) array[i] = i*10.0;

  /* Open viewer for binary output */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_WRITE,&view_out));
  CHKERRQ(PetscViewerBinaryGetDescriptor(view_out,&fd));

  /* Write binary output */
  CHKERRQ(PetscBinaryWrite(fd,&m,1,PETSC_INT));
  CHKERRQ(PetscBinaryWrite(fd,array,m,PETSC_SCALAR));

  /* Destroy the output viewer and work array */
  CHKERRQ(PetscViewerDestroy(&view_out));
  CHKERRQ(PetscFree(array));

  /* ---------------------------------------------------------------------- */
  /*          PART 2: Read data from file and form a vector                 */
  /* ---------------------------------------------------------------------- */

  /* Open input binary viewer */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_READ,&view_in));
  CHKERRQ(PetscViewerBinaryGetDescriptor(view_in,&fd));

  /* Create vector and get pointer to data space */
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&vec));
  CHKERRQ(VecSetSizes(vec,PETSC_DECIDE,m));
  CHKERRQ(VecSetFromOptions(vec));
  CHKERRQ(VecGetArray(vec,&avec));

  /* Read data into vector */
  CHKERRQ(PetscBinaryRead(fd,&sz,1,NULL,PETSC_INT));
  PetscCheckFalse(sz <=0,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Error: Must have array length > 0");

  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"reading data in binary from input.dat, sz =%" PetscInt_FMT " ...\n",sz));
  CHKERRQ(PetscBinaryRead(fd,avec,sz,NULL,PETSC_SCALAR));

  /* View vector */
  CHKERRQ(VecRestoreArray(vec,&avec));
  CHKERRQ(VecView(vec,PETSC_VIEWER_STDOUT_SELF));

  /* Free data structures */
  CHKERRQ(VecDestroy(&vec));
  CHKERRQ(PetscViewerDestroy(&view_in));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:

TEST*/
