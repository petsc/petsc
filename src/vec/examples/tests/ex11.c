

static char help[] = "Scatters from a parallel vector to a sequential vector.\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           ierr;
  int           numtids,mytid,i,N;
  Scalar        mone = -1.0, value;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0); 
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,mytid+1,-1,&x); CHKERRA(ierr);
  ierr = VecGetSize(x,&N);CHKERRA(ierr);
  ierr = VecCreateSequential(N-mytid,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSequential(N-mytid,mytid,1,&is1); CHKERRA(ierr);
  ierr = ISCreateStrideSequential(N-mytid,0,1,&is2); CHKERRA(ierr);

  /* fill parallel vector: note this is not efficient way*/
  for ( i=0; i<N; i++ ) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,InsertValues); CHKERRA(ierr);
  }
  ierr = VecBeginAssembly(x); CHKERRA(ierr);
  ierr = VecEndAssembly(x); CHKERRA(ierr);
  ierr = VecSet(&mone,y); CHKERRA(ierr);

  VecView(x,SYNC_STDOUT_VIEWER); 

  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,InsertValues,ScatterAll,ctx);
  CHKERRA(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,InsertValues,ScatterAll,ctx); CHKERRA(ierr);
  VecScatterCtxDestroy(ctx);
  

  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
