

/*
    Scatters from a parallel vector to a sequential vector.

*/
#include "petsc.h"
#include "comm.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr, idx2[3] = {0,2,3};
  int           numtids,mytid,i,N;
  Scalar        mone = -1.0, value;
  double        norm;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0); 
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,mytid+1,-1,&x); CHKERR(ierr);
  ierr = VecGetSize(x,&N);CHKERR(ierr);
  ierr = VecCreateSequential(N-mytid,&y); CHKERR(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSequential(N-mytid,mytid,1,&is1); CHKERR(ierr);
  ierr = ISCreateStrideSequential(N-mytid,0,1,&is2); CHKERR(ierr);

  /* fill parallel vector: note this is not efficient way*/
  for ( i=0; i<N; i++ ) {
    value = (Scalar) i;
    ierr = VecInsertValues(x,1,&i,&value); CHKERR(ierr);
  }
  ierr = VecBeginAssembly(x); CHKERR(ierr);
  ierr = VecEndAssembly(x); CHKERR(ierr);
  ierr = VecSet(&mone,y); CHKERR(ierr);

  VecView(x,0); 

  ierr = VecScatterBegin(x,is1,y,is2,&ctx); CHKERR(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,&ctx); CHKERR(ierr);
  VecScatterCtxDestroy(ctx);
  
  MPE_Seq_begin(MPI_COMM_WORLD,1);
  printf("-Node %d ---\n",mytid); VecView(y,0); fflush(stdout);
  MPE_Seq_end(MPI_COMM_WORLD,1);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);

  PetscFinalize(); 
  return 0;
}
 
