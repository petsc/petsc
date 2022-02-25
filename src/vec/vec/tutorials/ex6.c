
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  /* ---------------------------------------------------------------------- */
  /*          PART 1: Write some data to a file in binary format            */
  /* ---------------------------------------------------------------------- */

  /* Allocate array and set values */
  ierr = PetscMalloc1(m,&array);CHKERRQ(ierr);
  for (i=0; i<m; i++) array[i] = i*10.0;

  /* Open viewer for binary output */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(view_out,&fd);CHKERRQ(ierr);

  /* Write binary output */
  ierr = PetscBinaryWrite(fd,&m,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(fd,array,m,PETSC_SCALAR);CHKERRQ(ierr);

  /* Destroy the output viewer and work array */
  ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);
  ierr = PetscFree(array);CHKERRQ(ierr);

  /* ---------------------------------------------------------------------- */
  /*          PART 2: Read data from file and form a vector                 */
  /* ---------------------------------------------------------------------- */

  /* Open input binary viewer */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_READ,&view_in);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(view_in,&fd);CHKERRQ(ierr);

  /* Create vector and get pointer to data space */
  ierr = VecCreate(PETSC_COMM_SELF,&vec);CHKERRQ(ierr);
  ierr = VecSetSizes(vec,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vec);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);

  /* Read data into vector */
  ierr = PetscBinaryRead(fd,&sz,1,NULL,PETSC_INT);CHKERRQ(ierr);
  PetscCheckFalse(sz <=0,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Error: Must have array length > 0");

  ierr = PetscPrintf(PETSC_COMM_SELF,"reading data in binary from input.dat, sz =%" PetscInt_FMT " ...\n",sz);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,avec,sz,NULL,PETSC_SCALAR);CHKERRQ(ierr);

  /* View vector */
  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  ierr = VecView(vec,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /* Free data structures */
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view_in);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:

TEST*/
