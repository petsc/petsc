#ifndef lint
static char vcid[] = "$Id: ex23.c,v 1.4 1996/03/19 21:23:15 bsmith Exp $";
#endif

static char help[] = "Writes an array to a file, then reads an array from\n\
a file, then forms a vector.\n\n";

#include <stdio.h>
#include "vec.h"
#include "pinclude/pviewer.h"

int main(int argc,char **args)
{
  int     i, ierr, m = 10, flg, fd, size;
  Scalar  *avec, *array;
  Vec     vec;
  Viewer  view_out, view_in;

  PetscInitialize(&argc,&args,(char *)0,help);
  OptionsGetInt(PETSC_NULL,"-m",&m,&flg);

  /* ---------------------------------------------------------------------- */
  /*          PART 1: Write some data to a file in binary format            */
  /* ---------------------------------------------------------------------- */

  /* Allocate array and set values */
  array = (Scalar *) PetscMalloc( m*sizeof(Scalar) ); CHKPTRA(array);
  for (i=0; i<m; i++) {
    array[i] = i*10.0;
  }

  /* Open viewer for binary output */
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,"input.dat",BINARY_CREATE,&view_out);
         CHKERRA(ierr);
  ierr = ViewerBinaryGetDescriptor(view_out,&fd); CHKERRA(ierr);

  /* Write binary output */
  ierr = PetscBinaryWrite(fd,&m,1,BINARY_INT,0); CHKERRA(ierr);
  ierr = PetscBinaryWrite(fd,array,m,BINARY_SCALAR,0); CHKERRA(ierr);

  /* Destroy the output viewer and work array */
  ierr = ViewerDestroy(view_out); CHKERRA(ierr);
  PetscFree(array);

  /* ---------------------------------------------------------------------- */
  /*          PART 2: Read data from file and form a vector                 */
  /* ---------------------------------------------------------------------- */

  /* Open input binary viewer */
  ierr = ViewerFileOpenBinary(MPI_COMM_SELF,"input.dat",BINARY_RDONLY,&view_in); 
         CHKERRA(ierr);
  ierr = ViewerBinaryGetDescriptor(view_in,&fd); CHKERRA(ierr);

  /* Create vector and get pointer to data space */
  ierr = VecCreate(MPI_COMM_SELF,m,&vec); CHKERRA(ierr);
  ierr = VecGetArray(vec,&avec); CHKERRA(ierr);

  /* Read data into vector */
  ierr = PetscBinaryRead(fd,&size,1,BINARY_INT); CHKERRQ(ierr);
  if (size <=0) SETERRA(1,"Error: Must have array length > 0");

  printf("reading data in binary from input.dat, size =%d ...\n",size); 
  ierr = PetscBinaryRead(fd,avec,size,BINARY_SCALAR); CHKERRA(ierr);

  /* View vector */
  ierr = VecRestoreArray(vec,&avec); CHKERRA(ierr);
  ierr = VecView(vec,STDOUT_VIEWER_SELF); CHKERRA(ierr);

  /* Free data structures */
  ierr = VecDestroy(vec); CHKERRA(ierr);
  ierr = ViewerDestroy(view_in); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

