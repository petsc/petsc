

static char help[] = "Scatters from a parallel vector to a sequential vector.\n\
   Does case when we are merely selecting the local part of the\n\
   parallel vector.\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
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
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);

  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,-1,numtids*n,&x); CHKERRA(ierr);
  ierr = VecCreateSequential(n,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSequential(n,n*mytid,1,&is1); CHKERRA(ierr);
  ierr = ISCreateStrideSequential(n,0,1,&is2); CHKERRA(ierr);

  /* each processor inserts the entire vector */
  /* this is redundant but tests assembly */
  for ( i=0; i<n*numtids; i++ ) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,InsertValues); CHKERRA(ierr);
  }
  ierr = VecBeginAssembly(x); CHKERRA(ierr);
  ierr = VecEndAssembly(x); CHKERRA(ierr);

  VecView(x,SYNC_STDOUT_VIEWER); if (!mytid) printf("----\n");

  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,InsertValues,ScatterAll,ctx);
  CHKERRA(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,InsertValues,ScatterAll,ctx); CHKERRA(ierr);
  VecScatterCtxDestroy(ctx);
  
  if (!mytid) VecView(y,STDOUT_VIEWER);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(err);

  PetscFinalize(); 
  return 0;
}
 
