#ifndef lint
static char vcid[] = "$Id: ex12.c,v 1.31 1996/07/08 22:16:40 bsmith Exp bsmith $";
#endif

static char help[] = "Scatters from a sequential vector to a parallel vector.\n\
This does case when we are merely selecting the local part of the\n\
parallel vector.\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr, size,rank,i;
  Scalar        value;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help);

  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  /* create two vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,size*n,&x); CHKERRA(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStride(MPI_COMM_SELF,n,n*rank,1,&is1); CHKERRA(ierr);
  ierr = ISCreateStride(MPI_COMM_SELF,n,0,1,&is2); CHKERRA(ierr);

  /* each processor inserts the entire vector */
  /* this is redundant but tests assembly */
  for ( i=0; i<n; i++ ) {
    value = (Scalar) (i + 10*rank);
    ierr = VecSetValues(y,1,&i,&value,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(y); CHKERRA(ierr);
  ierr = VecAssemblyEnd(y); CHKERRA(ierr);

  ierr = VecScatterCreate(y,is2,x,is1,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(y,x,INSERT_VALUES,SCATTER_ALL,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(y,x,INSERT_VALUES,SCATTER_ALL,ctx); CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx); CHKERRA(ierr);
  
  ierr = VecView(x,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
