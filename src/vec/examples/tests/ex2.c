

/*
      Example demonstrating some features of the vectors directory.
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
  int           n = 5, ierr, idx1[2] = {0,3}, idx2[2] = {1,4};
  Scalar        one = 1.0, two = 2.0, three = 3.0, dots[3],dot;
  double        norm;
  Vec           x,y,w,*z;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  OptionsCreate(&argc,&argv,(char*)0,(char*)0);
  OptionsGetInt(0,"-n",&n);

  /* create two vector */
  ierr = VecCreateSequential(n,&x); CHKERR(ierr);
  ierr = VecCreate(x,&y); CHKERR(ierr);

  /* create two index sets */
  ierr = ISCreateSequential(2,idx1,&is1); CHKERR(ierr);
  ierr = ISCreateSequential(2,idx2,&is2); CHKERR(ierr);


  ierr = VecSet(&one,x);CHKERR(ierr);
  ierr = VecSet(&two,y);CHKERR(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,InsertValues,&ctx); CHKERR(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,InsertValues,&ctx); CHKERR(ierr);
  
  VecView(y,0);

  ierr = VecScatterBegin(y,is1,x,is2,InsertValues,&ctx); CHKERR(ierr);
  ierr = VecScatterEnd(y,is1,x,is2,InsertValues,&ctx); CHKERR(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERR(ierr);

  printf("-------\n");VecView(x,0);

  ierr = ISDestroy(is1); CHKERR(ierr);
  ierr = ISDestroy(is2); CHKERR(ierr);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);

  PetscFinalize();
  return 0;
}
 
