

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
  int           n = 5, ierr, idx2[3] = {0,2,3}, idx1[3] = {0,1,2};
  int           numtids,mytid,i;
  Scalar        mone = -1.0, value;
  double        norm;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0); 
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,-1,numtids*n,&x); CHKERR(ierr);
  ierr = VecCreateSequential(n,&y); CHKERR(ierr);

  /* create two index sets */
  ierr = ISCreateSequential(3,idx1,&is1); CHKERR(ierr);
  ierr = ISCreateSequential(3,idx2,&is2); CHKERR(ierr);

  /* fill local part of parallel vector */
  for ( i=n*mytid; i<n*(mytid+1); i++ ) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,InsertValues); CHKERR(ierr);
  }
  ierr = VecBeginAssembly(x); CHKERR(ierr);
  ierr = VecEndAssembly(x); CHKERR(ierr);
  ierr = VecSet(&mone,y); CHKERR(ierr);

  VecView(x,0); printf("----\n");

  ierr = VecScatterBegin(x,is1,y,is2,InsertValues,&ctx); CHKERR(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,InsertValues,&ctx); CHKERR(ierr);
  VecScatterCtxDestroy(ctx);
  
  VecView(y,0);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);

  PetscFinalize(); 
  return 0;
}
 
