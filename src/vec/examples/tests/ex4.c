

static char help[] = "Scatters from parallel vector into seqential vectors.\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr, idx1[2] = {0,3}, idx2[2] = {1,4};
  Scalar        one = 1.0, two = 2.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-n",&n);

  /* create two vector */
  ierr = VecCreateMPI(MPI_COMM_WORLD,n,-1,&x); CHKERR(ierr);
  ierr = VecCreateSequential(n,&y); CHKERR(ierr);

  /* create two index sets */
  ierr = ISCreateSequential(2,idx1,&is1); CHKERR(ierr);
  ierr = ISCreateSequential(2,idx2,&is2); CHKERR(ierr);


  ierr = VecSet(&one,x);CHKERR(ierr);
  ierr = VecSet(&two,y);CHKERR(ierr);
  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERR(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,InsertValues,ScatterAll,ctx);
  CHKERR(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,InsertValues,ScatterAll,ctx); CHKERR(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERR(ierr);
  
  VecView(y,0);

  ierr = ISDestroy(is1); CHKERR(ierr);
  ierr = ISDestroy(is2); CHKERR(ierr);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);
  PetscFinalize();

  return 0;
}
 
