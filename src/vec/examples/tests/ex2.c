
static char help[] = "Tests vector scatter gathers\n";

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
  ierr = VecCreateSequential(n,&x); CHKERR(ierr);
  ierr = VecCreate(x,&y); CHKERR(ierr);

  /* create two index sets */
  ierr = ISCreateSequential(2,idx1,&is1); CHKERR(ierr);
  ierr = ISCreateSequential(2,idx2,&is2); CHKERR(ierr);


  ierr = VecSet(&one,x);CHKERR(ierr);
  ierr = VecSet(&two,y);CHKERR(ierr);
  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERR(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,InsertValues,ScatterAll,ctx);
  CHKERR(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,InsertValues,ScatterAll,ctx); CHKERR(ierr);
  
  VecView(y,STDOUT_VIEWER);

  ierr = VecScatterBegin(y,is1,x,is2,InsertValues,ScatterAll,ctx);
  CHKERR(ierr);
  ierr = VecScatterEnd(y,is1,x,is2,InsertValues,ScatterAll,ctx); CHKERR(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERR(ierr);

  printf("-------\n");VecView(x,STDOUT_VIEWER);

  ierr = ISDestroy(is1); CHKERR(ierr);
  ierr = ISDestroy(is2); CHKERR(ierr);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);

  PetscFinalize();
  return 0;
}
 
