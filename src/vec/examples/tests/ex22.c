#ifndef lint
static char vcid[] = "$Id: ex22.c,v 1.4 1996/07/08 22:16:40 bsmith Exp bsmith $";
#endif

static char help[] = "Scatters from a parallel vector to a parallel vector.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr;
  int           size,rank,N;
  Scalar        value,zero = 0.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  /* create two vectors */
  N = size*n;
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,N,&y); CHKERRA(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,N,&x); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStride(MPI_COMM_SELF,2*n,0,1,&is1); CHKERRA(ierr);
  ierr = ISCreateStride(MPI_COMM_SELF,2*n,n+2,1,&is2); CHKERRA(ierr);

  value = rank+1; 
  ierr = VecSet(&value,x); CHKERRA(ierr);
  ierr = VecSet(&zero,y); CHKERRA(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,ADD_VALUES,SCATTER_ALL,ctx); CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,ADD_VALUES,SCATTER_ALL,ctx); CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx); CHKERRA(ierr);
  
  ierr = VecView(y,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
