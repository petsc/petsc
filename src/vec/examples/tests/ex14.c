

static char help[] = "Scatters from  a sequential vector to a parallel vector.\n\
   Does tricky case.\n";

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
  int           numtids,mytid,N;
  Scalar        value;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  N = numtids*n;
  ierr = VecCreateMPI(MPI_COMM_WORLD,-1,N,&y); CHKERR(ierr);
  ierr = VecCreateSequential(N,&x); CHKERR(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSequential(n,0,1,&is1); CHKERR(ierr);
  ierr = ISCreateStrideSequential(n,mytid,1,&is2); CHKERR(ierr);

  value = mytid+1; VecSet(&value,x);

  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERR(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,AddValues,ScatterAll,ctx); CHKERR(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,AddValues,ScatterAll,ctx); CHKERR(ierr);
  VecScatterCtxDestroy(ctx);
  
  VecView(y,SYNC_STDOUT_VIEWER);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);
  ierr = ISDestroy(is1);CHKERR(ierr);
  ierr = ISDestroy(is2);CHKERR(ierr);

  PetscFinalize(); 
  return 0;
}
 
