#ifndef lint
static char vcid[] = "$Id: ex20.c,v 1.2 1995/08/17 23:42:52 curfman Exp curfman $";
#endif

static char help[] = 
"This example tests binary I/O of vectors and illustrates the use of\n\
user-defined event logging.\n\n";

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "vec.h"
#include "plog.h"

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output. */

int main(int argc,char **args)
{
  int     i, lm, m = 10, mytid, numtids, low, high, ldim, iglobal, ierr, fd;
  Scalar  v;
  Vec     u;
  IS      ind;
  char    filename[10];
  VecType vtype;

#define VECTOR_GENERATE 76
#define VECTOR_READ     77

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
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
    ierr = VecSetValues(u,1,&iglobal,&v,INSERTVALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(u); CHKERRA(ierr);
  ierr = VecAssemblyEnd(u); CHKERRA(ierr);
  ierr = VecView(u,STDOUT_VIEWER); CHKERRA(ierr);

  MPIU_printf(MPI_COMM_WORLD,"writing vector in binary to vector.dat ...\n"); 
  sprintf(filename,"vector.dat");
  if ((fd = creat(filename, 0666)) == -1)
    SETERRA(1,"Cannot create filename for writing.");
  ierr = VecViewBinary(u,fd); CHKERRA(ierr);
  close(fd);
  ierr = VecDestroy(u); CHKERRA(ierr);
  PLogEventEnd(VECTOR_GENERATE,0,0,0,0);

  /* PART 2:  Read in vector in binary format */

  /* All processors wait until test vector has been dumped */
  MPI_Barrier(MPI_COMM_WORLD);

  /* lm = number of locally owned vector elements */
  lm = m/numtids + ((m % numtids) > mytid);
  ierr = ISCreateStrideSequential(MPI_COMM_WORLD,lm,mytid,numtids,&ind);
  CHKERRA(ierr);

  /* Read new vector in binary format */
  PLogEventRegister(VECTOR_READ,"Read Vector     ");
  PLogEventBegin(VECTOR_READ,0,0,0,0);
  MPIU_printf(MPI_COMM_WORLD,"reading vector in binary from vector.dat ...\n"); 
  sprintf(filename,"vector.dat");
  if ((fd = open(filename, O_RDONLY, 0)) == -1) {
    SETERRQ(1,"Cannot open filename for reading.");
  }
  vtype = VECSEQ;
  if (OptionsHasName(0,"-mpi_objects") || numtids>1) vtype = VECMPI;
  VecLoadBinary(MPI_COMM_WORLD,fd,vtype,ind,&u); CHKERRA(ierr);
  close(fd);
  PLogEventEnd(VECTOR_READ,0,0,0,0);
  VecView(u,SYNC_STDOUT_VIEWER); CHKERRA(ierr);

  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = ISDestroy(ind); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


