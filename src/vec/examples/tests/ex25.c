#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex25.c,v 1.3 1998/12/03 03:57:07 bsmith Exp bsmith $";
#endif

static char help[] = "Scatters from a parallel vector to a sequential vector.  In\n\
this case processor zero is as long as the entire parallel vector; rest are zero length.\n\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5, ierr;
  int           size,rank,N,low,high,iglobal,i;
  Scalar        value,zero = 0.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  /* create two vectors */
  N = size*n;
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,N,&y); CHKERRA(ierr);
  if (!rank) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x); CHKERRA(ierr);
  } else {
    ierr = VecCreateSeq(PETSC_COMM_SELF,0,&x); CHKERRA(ierr);
  }

  /* create two index sets */
  if (!rank) {
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is1); CHKERRA(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is2); CHKERRA(ierr);
  } else {
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is1); CHKERRA(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is2); CHKERRA(ierr);
  }

  ierr = VecSet(&zero,x); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(y,&low,&high); CHKERRA(ierr);
  for ( i=0; i<n; i++ ) {
    iglobal = i + low; value = (Scalar) (i + 10*rank);
    ierr = VecSetValues(y,1,&iglobal,&value,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(y); CHKERRA(ierr);
  ierr = VecAssemblyEnd(y); CHKERRA(ierr);
  ierr = VecView(y,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = VecScatterCreate(y,is2,x,is1,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(y,x,ADD_VALUES,SCATTER_FORWARD,ctx); CHKERRA(ierr);
  ierr = VecScatterEnd(y,x,ADD_VALUES,SCATTER_FORWARD,ctx); CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx); CHKERRA(ierr);
  
  if (!rank) 
    {printf("----\n"); ierr = VecView(x,VIEWER_STDOUT_SELF); CHKERRA(ierr);}

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
