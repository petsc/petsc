
static char help[] = "Writes an array to a file, then reads an array from a file, then forms a vector.\n\n";

#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  int            fd;
  PetscInt       i,m = 10,sz;
  PetscScalar    *avec,*array;
  Vec            vec;
  PetscViewer    view_out,view_in;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(1,"This is a uniprocessor example only!");
  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);

  /* ---------------------------------------------------------------------- */
  /*          PART 1: Write some data to a file in binary format            */
  /* ---------------------------------------------------------------------- */

  /* Allocate array and set values */
  ierr = PetscMalloc(m*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    array[i] = i*10.0;
  }

  /* Open viewer for binary output */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_WRITE,&view_out);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(view_out,&fd);CHKERRQ(ierr);

  /* Write binary output */
  ierr = PetscBinaryWrite(fd,&m,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(fd,array,m,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);

  /* Destroy the output viewer and work array */
  ierr = PetscViewerDestroy(view_out);CHKERRQ(ierr);
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
  ierr = PetscBinaryRead(fd,&sz,1,PETSC_INT);CHKERRQ(ierr);
  if (sz <=0) SETERRQ(1,"Error: Must have array length > 0");

  ierr = PetscPrintf(PETSC_COMM_SELF,"reading data in binary from input.dat, sz =%D ...\n",sz);CHKERRQ(ierr); 
  ierr = PetscBinaryRead(fd,avec,sz,PETSC_SCALAR);CHKERRQ(ierr);

  /* View vector */
  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  ierr = VecView(vec,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /* Free data structures */
  ierr = VecDestroy(vec);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(view_in);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

