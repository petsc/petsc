#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex5.c,v 1.35 1999/04/21 18:16:01 bsmith Exp balay $";
#endif

static char help[] = "Tests binary I/O of vectors and illustrates the use of\n\
user-defined event logging.\n\n";

#include "vec.h"

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output. */

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int     i, m = 10, rank, size, low, high, ldim, iglobal, ierr,flg;
  Scalar  v;
  Vec     u;
  Viewer  viewer;
  int     VECTOR_GENERATE, VECTOR_READ;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);

  /* PART 1:  Generate vector, then write it in binary format */

  ierr = PLogEventRegister(&VECTOR_GENERATE,"Generate Vector","Red:");CHKERRA(ierr);
  PLogEventBegin(VECTOR_GENERATE,0,0,0,0);
  /* Generate vector */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m,&u);CHKERRA(ierr);
  ierr = VecSetFromOptions(u);CHKERRA(ierr);
  ierr = VecGetOwnershipRange(u,&low,&high);CHKERRA(ierr);
  ierr = VecGetLocalSize(u,&ldim);CHKERRA(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = (Scalar)(i + 100*rank);
    ierr = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRA(ierr);
  ierr = VecAssemblyEnd(u);CHKERRA(ierr);
  ierr = VecView(u,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"writing vector in binary to vector.dat ...\n"); 

  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",BINARY_CREATE,&viewer);CHKERRA(ierr);
  ierr = VecView(u,viewer);CHKERRA(ierr);
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  PLogEventEnd(VECTOR_GENERATE,0,0,0,0);

  /* PART 2:  Read in vector in binary format */

  /* All processors wait until test vector has been dumped */
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscSleep(10);

  /* Read new vector in binary format */
  ierr = PLogEventRegister(&VECTOR_READ,"Read Vector","Green:");CHKERRA(ierr);
  PLogEventBegin(VECTOR_READ,0,0,0,0);
  PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from vector.dat ...\n"); 
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",BINARY_RDONLY,&viewer);CHKERRA(ierr);
  ierr = VecLoad(viewer,&u);CHKERRA(ierr);
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  PLogEventEnd(VECTOR_READ,0,0,0,0);
  ierr = VecView(u,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Free data structures */
  ierr = VecDestroy(u);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

