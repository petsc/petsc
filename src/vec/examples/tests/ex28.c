#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex28.c,v 1.5 1999/03/28 19:18:54 bsmith Exp bsmith $";
#endif

static char help[] = "Tests repeated VecDotBegin()/VecDotEnd()\n\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           ierr, n = 25,i,row0 = 0;
  Scalar        one = 1.0, two = 2.0,result1,result2,results[40],value,ten = 10.0;
  double        result3,result4,result[2];
  Vec           x,y,vecs[40];

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* create vector */
  ierr = VecCreate(PETSC_COMM_WORLD,n,PETSC_DECIDE,&x); CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);

  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecSet(&two,y); CHKERRA(ierr);

  /*
        Test mixing dot products and norms that require sums
  */
  result1 = result2 = 0.0;
  result3 = result4 = 0.0;
  ierr = VecDotBegin(x,y,&result1);CHKERRA(ierr);
  ierr = VecDotBegin(y,x,&result2);CHKERRA(ierr);
  ierr = VecNormBegin(y,NORM_2,&result3);CHKERRA(ierr);
  ierr = VecNormBegin(x,NORM_1,&result4);CHKERRA(ierr);
  ierr = VecDotEnd(x,y,&result1);CHKERRA(ierr);
  ierr = VecDotEnd(y,x,&result2);CHKERRA(ierr);
  ierr = VecNormEnd(y,NORM_2,&result3);CHKERRA(ierr);
  ierr = VecNormEnd(x,NORM_1,&result4);CHKERRA(ierr);
  if (result1 != 50.0 || result2 != 50.0) {
    PetscPrintf(PETSC_COMM_WORLD,"Error dot: result1 %g result2 %g\n",PetscReal(result1),PetscReal(result2));
  }
  if (result3 != 10.0 || result4 != 25.0) {
    PetscPrintf(PETSC_COMM_WORLD,"Error 1,2 norms: result3 %g result4 %g\n",result3,result4);
  }

  /*
        Test norms that only require abs
  */
  result1 = result2 = 0.0;
  result3 = result4 = 0.0;
  ierr = VecNormBegin(y,NORM_MAX,&result3);CHKERRA(ierr);
  ierr = VecNormBegin(x,NORM_MAX,&result4);CHKERRA(ierr);
  ierr = VecNormEnd(y,NORM_MAX,&result3);CHKERRA(ierr);
  ierr = VecNormEnd(x,NORM_MAX,&result4);CHKERRA(ierr);
  if (result3 != 2.0 || result4 != 1.0) {
    PetscPrintf(PETSC_COMM_WORLD,"Error max norm: result3 %g result4 %g\n",result3,result4);
  }

  /*
        Tests dot,  max, 1, norm
  */
  result1 = result2 = 0.0;
  result3 = result4 = 0.0;
  ierr = VecSetValues(x,1,&row0,&ten,INSERT_VALUES);CHKERRA(ierr);
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);
  ierr = VecDotBegin(x,y,&result1);CHKERRA(ierr);
  ierr = VecDotBegin(y,x,&result2);CHKERRA(ierr);
  ierr = VecNormBegin(x,NORM_MAX,&result3);CHKERRA(ierr);
  ierr = VecNormBegin(x,NORM_1,&result4);CHKERRA(ierr);
  ierr = VecDotEnd(x,y,&result1);CHKERRA(ierr);
  ierr = VecDotEnd(y,x,&result2);CHKERRA(ierr);
  ierr = VecNormEnd(x,NORM_MAX,&result3);CHKERRA(ierr);
  ierr = VecNormEnd(x,NORM_1,&result4);CHKERRA(ierr);
  if (result1 != 68.0 || result2 != 68.0) {
    PetscPrintf(PETSC_COMM_WORLD,"Error dot: result1 %g result2 %g\n",PetscReal(result1),PetscReal(result2));
  }
  if (result3 != 10.0 || result4 != 34.0) {
    PetscPrintf(PETSC_COMM_WORLD,"Error max 1 norms: result3 %g result4 %g\n",result3,result4);
  }

  /*
       tests 1_and_2 norm 
  */
  ierr = VecNormBegin(x,NORM_MAX,&result3);CHKERRA(ierr);
  ierr = VecNormBegin(x,NORM_1_AND_2,result);CHKERRA(ierr);
  ierr = VecNormBegin(y,NORM_MAX,&result4);CHKERRA(ierr);
  ierr = VecNormEnd(x,NORM_MAX,&result3);CHKERRA(ierr);
  ierr = VecNormEnd(x,NORM_1_AND_2,result);CHKERRA(ierr);
  ierr = VecNormEnd(y,NORM_MAX,&result4);CHKERRA(ierr);
  if (result3 != 10.0 || result4 != 2.0) {
    PetscPrintf(PETSC_COMM_WORLD,"Error max: result1 %g result2 %g\n",result3,result4);
  }
  if (PetscAbsDouble(result[0]-34.0) > .01 || PetscAbsDouble(result[1]-sqrt(124.0)) > .01) {
    PetscPrintf(PETSC_COMM_WORLD,"Error 1 and 2 norms: result[0] %g result[1] %g\n",result[0],result[1]);
  }

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  /*
       Tests computing a large number of operations that require 
    allocating a larger data structure internally
  */
  for (i=0; i<40; i++) {
    ierr  = VecCreate(PETSC_COMM_SELF,n,PETSC_DECIDE,vecs+i); CHKERRA(ierr);
    ierr  = VecSetFromOptions(vecs[i]);CHKERRA(ierr);
    value = (double) i;
    ierr  = VecSet(&value,vecs[i]);CHKERRA(ierr);
  }
  for (i=0; i<39; i++) {
    ierr = VecDotBegin(vecs[i],vecs[i+1],results+i);CHKERRA(ierr);
  }
  for (i=0; i<39; i++) {
    ierr = VecDotEnd(vecs[i],vecs[i+1],results+i);CHKERRA(ierr);
    if (results[i] != 25.0*i*(i+1)) {
      PetscPrintf(PETSC_COMM_WORLD,"i %d expected %g got %g\n",i,25.0*i*(i+1),PetscReal(results[i]));
    }
  } 
  for (i=0; i<40; i++) {
    ierr = VecDestroy(vecs[i]);CHKERRA(ierr);
  }

  PetscFinalize();
  return 0;
}
 



