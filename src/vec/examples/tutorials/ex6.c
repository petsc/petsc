/*$Id: ex6.c,v 1.26 2001/01/15 21:45:20 bsmith Exp balay $*/

static char help[] = "Writes an array to a file, then reads an array from\n\
a file, then forms a vector.\n\n";

#include "petscvec.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int     i,ierr,m = 10,fd,size,sz;
  Scalar  *avec,*array;
  Vec     vec;
  PetscViewer  view_out,view_in;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&sz);CHKERRA(ierr);
  if (sz != 1) SETERRA(1,"This is a uniprocessor example only!");
  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);

  /* ---------------------------------------------------------------------- */
  /*          PART 1: Write some data to a file in binary format            */
  /* ---------------------------------------------------------------------- */

  /* Allocate array and set values */
  ierr = PetscMalloc(m*sizeof(Scalar),&array);CHKERRA(ierr);
  for (i=0; i<m; i++) {
    array[i] = i*10.0;
  }

  /* Open viewer for binary output */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",PETSC_BINARY_CREATE,&view_out);CHKERRA(ierr);
  ierr = PetscViewerBinaryGetDescriptor(view_out,&fd);CHKERRA(ierr);

  /* Write binary output */
  ierr = PetscBinaryWrite(fd,&m,1,PETSC_INT,0);CHKERRA(ierr);
  ierr = PetscBinaryWrite(fd,array,m,PETSC_SCALAR,0);CHKERRA(ierr);

  /* Destroy the output viewer and work array */
  ierr = PetscViewerDestroy(view_out);CHKERRA(ierr);
  ierr = PetscFree(array);CHKERRA(ierr);

  /* ---------------------------------------------------------------------- */
  /*          PART 2: Read data from file and form a vector                 */
  /* ---------------------------------------------------------------------- */

  /* Open input binary viewer */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",PETSC_BINARY_RDONLY,&view_in);CHKERRA(ierr);
  ierr = PetscViewerBinaryGetDescriptor(view_in,&fd);CHKERRA(ierr);

  /* Create vector and get pointer to data space */
  ierr = VecCreate(PETSC_COMM_SELF,PETSC_DECIDE,m,&vec);CHKERRA(ierr);
  ierr = VecSetFromOptions(vec);CHKERRA(ierr);
  ierr = VecGetArray(vec,&avec);CHKERRA(ierr);

  /* Read data into vector */
  ierr = PetscBinaryRead(fd,&size,1,PETSC_INT);CHKERRQ(ierr);
  if (size <=0) SETERRA(1,"Error: Must have array length > 0");

  ierr = PetscPrintf(PETSC_COMM_SELF,"reading data in binary from input.dat, size =%d ...\n",size);CHKERRA(ierr); 
  ierr = PetscBinaryRead(fd,avec,size,PETSC_SCALAR);CHKERRA(ierr);

  /* View vector */
  ierr = VecRestoreArray(vec,&avec);CHKERRA(ierr);
  ierr = VecView(vec,PETSC_VIEWER_STDOUT_SELF);CHKERRA(ierr);

  /* Free data structures */
  ierr = VecDestroy(vec);CHKERRA(ierr);
  ierr = PetscViewerDestroy(view_in);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

