#ifndef lint
static char vcid[] = "$Id: ex17.c,v 1.13 1995/10/12 04:13:20 bsmith Exp curfman $";
#endif

static char help[] = "Scatters from a parallel vector to a sequential vector.  In\n\
this case each local vector is as long as the entire parallel vector.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr;
  int           size,rank,N,low,high,iglobal,i;
  Scalar        value,zero = 0.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  /* create two vectors */
  N = size*n;
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,N,&y); CHKERRA(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&x); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,N,0,1,&is1); CHKERRA(ierr);
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,N,0,1,&is2); CHKERRA(ierr);

  ierr = VecSet(&zero,x); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(y,&low,&high); CHKERRA(ierr);
  for ( i=0; i<n; i++ ) {
    iglobal = i + low; value = (Scalar) (i + 10*rank);
    ierr = VecSetValues(y,1,&iglobal,&value,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(y); CHKERRA(ierr);
  ierr = VecAssemblyEnd(y); CHKERRA(ierr);
  ierr = VecView(y,STDOUT_VIEWER_WORLD); CHKERRA(ierr);

  ierr = VecScatterCtxCreate(y,is2,x,is1,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(y,x,ADD_VALUES,SCATTER_ALL,ctx); CHKERRA(ierr);
  ierr = VecScatterEnd(y,x,ADD_VALUES,SCATTER_ALL,ctx); CHKERRA(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERRA(ierr);
  
  if (!rank) 
    {printf("----\n"); ierr = VecView(x,STDOUT_VIEWER_SELF); CHKERRA(ierr);}

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
