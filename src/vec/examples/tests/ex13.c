#ifndef lint
static char vcid[] = "$Id: ex13.c,v 1.20 1995/08/17 21:33:33 curfman Exp curfman $";
#endif

static char help[] = 
"This example scatters from a sequential vector to a parallel vector.  In\n\
this case each local vector is as long as the entire parallel vector.\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr;
  int           numtids,mytid,i,N;
  Scalar        value;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  N = numtids*n;
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,N,&y); CHKERRA(ierr);
  ierr = VecCreateSequential(MPI_COMM_SELF,N,&x); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,N,0,1,&is1); CHKERRA(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,N,0,1,&is2); CHKERRA(ierr);

  for ( i=0; i<N; i++ ) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,INSERTVALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  ierr = VecScatterCtxCreate(x,is2,y,is1,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERTVALUES,SCATTERALL,ctx); CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERTVALUES,SCATTERALL,ctx); CHKERRA(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERRA(ierr);
  
  ierr = VecView(y,STDOUT_VIEWER_COMM); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
