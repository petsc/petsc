

/*
    Demonstrates a scatter with a stride and general index set
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
  int           n = 6, ierr, idx1[3] = {0,1,2}, loc[6] = {0,1,2,3,4,5};
  Scalar        one = 1.0, two = 2.0, vals[6] = {10,11,12,13,14,15};
  double        norm;
  Vec           x,y,w,*z;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  OptionsCreate(argc,argv,(char*)0,(char*)0);

  /* create two vector */
  ierr = VecCreateSequential(n,&x); CHKERR(ierr);
  ierr = VecCreate(x,&y); CHKERR(ierr);

  /* create two index sets */
  ierr = ISCreateSequential(3,idx1,&is1); CHKERR(ierr);
  ierr = ISCreateStrideSequential(3,0,2,&is2); CHKERR(ierr);

  ierr = VecInsertValues(x,6,loc,vals); CHKERR(ierr);
  VecView(x,0); printf("----\n");
  ierr = VecSet(&two,y);CHKERR(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,&ctx); CHKERR(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,&ctx); CHKERR(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERR(ierr);
  
  VecView(y,0);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);

  PetscFinalize();
  return 0;
}
 
