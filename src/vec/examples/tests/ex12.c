#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.22 1995/08/17 14:11:05 curfman Exp $";
#endif

static char help[] = 
"This example scatters from a sequential vector to a parallel vector.\n\
This does case when we are merely selecting the local part of the\n\
parallel vector.\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr;
  int           numtids,mytid,i;
  Scalar        value;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);

  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,numtids*n,&x); CHKERRA(ierr);
  ierr = VecCreateSequential(MPI_COMM_SELF,n,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,n,n*mytid,1,&is1); 
  CHKERRA(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,n,0,1,&is2); 
  CHKERRA(ierr);

  /* each processor inserts the entire vector */
  /* this is redundant but tests assembly */
  for ( i=0; i<n; i++ ) {
    value = (Scalar) (i + 10*mytid);
    ierr = VecSetValues(y,1,&i,&value,INSERTVALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(y); CHKERRA(ierr);
  ierr = VecAssemblyEnd(y); CHKERRA(ierr);

  ierr = VecScatterCtxCreate(y,is2,x,is1,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(y,x,INSERTVALUES,SCATTERALL,ctx);
  CHKERRA(ierr);
  ierr = VecScatterEnd(y,x,INSERTVALUES,SCATTERALL,ctx); CHKERRA(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERRA(ierr);
  
  ierr = VecView(x,SYNC_STDOUT_VIEWER); CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
