#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex13.c,v 1.39 1999/04/19 22:11:24 bsmith Exp balay $";
#endif

static char help[] = "Scatters from a sequential vector to a parallel vector.  In\n\
this case each local vector is as long as the entire parallel vector.\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5, ierr;
  int           size,rank,i,N;
  Scalar        value;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  /* create two vectors */
  N = size*n;
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,N,&y);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is1);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is2);CHKERRA(ierr);

  for ( i=0; i<N; i++ ) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  ierr = VecScatterCreate(x,is2,y,is1,&ctx);CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRA(ierr);
  
  ierr = VecView(y,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
