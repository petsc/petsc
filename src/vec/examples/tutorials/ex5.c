#ifndef lint
static char vcid[] = "$Id: ex20.c,v 1.9 1995/09/21 20:08:16 bsmith Exp bsmith $";
#endif

static char help[] = 
"This example tests binary I/O of vectors and illustrates the use of\n\
user-defined event logging.\n\n";

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "vec.h"

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output. */

int main(int argc,char **args)
{
  int     i, m = 10, mytid, numtids, low, high, ldim, iglobal, ierr;
  Scalar  v;
  Vec     u;
  VecType vtype;
  Viewer  bview;

#define VECTOR_GENERATE 76
#define VECTOR_READ     77

  PetscInitialize(&argc,&args,0,0,help);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  OptionsGetInt(0,"-m",&m);

  /* PART 1:  Generate vector, then write it in binary format */

  PLogEventRegister(VECTOR_GENERATE,"Generate Vector ");
  PLogEventBegin(VECTOR_GENERATE,0,0,0,0);
  /* Generate vector */
  ierr = VecCreate(MPI_COMM_WORLD,m,&u); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(u,&low,&high); CHKERRA(ierr);
  ierr = VecGetLocalSize(u,&ldim); CHKERRA(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = (Scalar)(i + 100*mytid);
    ierr = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(u); CHKERRA(ierr);
  ierr = VecAssemblyEnd(u); CHKERRA(ierr);
  ierr = VecView(u,STDOUT_VIEWER_WORLD); CHKERRA(ierr);

  MPIU_printf(MPI_COMM_WORLD,"writing vector in binary to vector.dat ...\n"); 

  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,"vector.dat",BINARY_CREATE,&bview); 
         CHKERRA(ierr);
  ierr = VecView(u,bview); CHKERRA(ierr);
  ierr = ViewerDestroy(bview); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  PLogEventEnd(VECTOR_GENERATE,0,0,0,0);

  /* PART 2:  Read in vector in binary format */

  /* All processors wait until test vector has been dumped */
  MPI_Barrier(MPI_COMM_WORLD);
  PetscSleep(10);

  CHKERRA(ierr);

  /* Read new vector in binary format */
  PLogEventRegister(VECTOR_READ,"Read Vector     ");
  PLogEventBegin(VECTOR_READ,0,0,0,0);
  MPIU_printf(MPI_COMM_WORLD,"reading vector in binary from vector.dat ...\n"); 
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,"vector.dat",BINARY_RDONLY,&bview); 
         CHKERRA(ierr);
  vtype = VECSEQ;
  ierr = VecLoad(bview,&u); CHKERRA(ierr);
  ierr = ViewerDestroy(bview); CHKERRA(ierr);
  PLogEventEnd(VECTOR_READ,0,0,0,0);
  ierr = VecView(u,STDOUT_VIEWER_WORLD); CHKERRA(ierr);

  /* Free data structures */
  ierr = VecDestroy(u); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

