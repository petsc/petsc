#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.22 1995/08/17 14:11:05 curfman Exp $";
#endif

static char help[] = 
"This example tests binary I/O of vectors.\n\n";

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "vec.h"
#include "plog.h"

int main(int argc,char **args)
{
  Scalar  v, none = -1.0, norm;
  int     I, J, ldim, ierr, fd;
  int     i, j, m = 10, mytid, numtids, its;
  Vec     u, x, b;
  IS      ind;
  char    filename[128];
  VecType vtype;
  Viewer  viewer, viewer2;

#define VECTOR_GENERATE 76
#define VECTOR_READ     77

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  OptionsGetInt(0,"-m",&m);

  PLogEventRegister(VECTOR_GENERATE,"Generate Vector ");
  PLogEventBegin(VECTOR_GENERATE,0,0,0,0);
  if (!mytid) {
    /* Generate vector */
    ierr = VecCreateSequential(MPI_COMM_SELF,m,&u); CHKERRA(ierr);
    ierr = VecGetSize(u,&ldim); CHKERRA(ierr);
    for (i=0; i<ldim; i++) {
      v = (Scalar)(i + 100*mytid);
      ierr = VecSetValues(u,1,&i,&v,INSERTVALUES); CHKERRA(ierr);
    }
    ierr = VecAssemblyBegin(u); CHKERRA(ierr);
    ierr = VecAssemblyEnd(u); CHKERRA(ierr);

    ViewerFileOpen("vector.1",&viewer);
    ierr = VecView(u,viewer); CHKERRA(ierr);

    printf("dumping vector\n"); 
    sprintf(filename,"vector.dat");
    if ((fd = creat(filename, 0666)) == -1)
      SETERRA(1,"Cannot create filename for writing.");
    ierr = VecViewBinary(u,fd); CHKERRA(ierr);
    close(fd);

    ierr = ViewerDestroy(viewer); CHKERRA(ierr);
    ierr = VecDestroy(u); CHKERRA(ierr);
  }
    PLogEventEnd(VECTOR_GENERATE,0,0,0,0);

  /* All processors wait until test matrix and vector have been dumped */
  MPI_Barrier(MPI_COMM_WORLD);

  /* Find owned grid points (corresponding to matrix rows) */
   ierr = ISCreateStrideSequential(MPI_COMM_WORLD,

  /* Read new vector in binary format */
  PLogEventRegister(VECTOR_READ,"Read Vector     ");
  PLogEventBegin(VECTOR_READ,0,0,0,0);
  sprintf(filename,"vector.dat");
  if ((fd = open(filename, O_RDONLY, 0)) == -1) {
    SETERRQ(1,"Cannot open filename for reading.");
  }
  vtype = VECSEQ;
  if (OptionsHasName(0,"-mpi_objects")) vtype = VECMPI;
  VecLoadBinary(MPI_COMM_WORLD,fd,vtype,ind,&u); CHKERRA(ierr);
  close(fd);
  PLogEventEnd(VECTOR_READ,0,0,0,0);
  ViewerFileOpenSync("vector.2",MPI_COMM_WORLD,&viewer);
  VecView(u,viewer); CHKERRA(ierr);

  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}


