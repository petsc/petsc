
static char help[] = "Tests relaxation for dense matrices.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  Vec            u,x,b,e;
  PetscInt       i,n = 10,midx[3];
  PetscScalar    v[3];
  PetscReal      omega = 1.0,norm;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-omega",&omega,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&C));
  CHKERRQ(MatSetSizes(C,n,n,n,n));
  CHKERRQ(MatSetType(C,MATSEQDENSE));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&b));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&u));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&e));
  CHKERRQ(VecSet(u,1.0));
  CHKERRQ(VecSet(x,0.0));

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for (i=1; i<n-1; i++) {
    midx[0] = i-1; midx[1] = i; midx[2] = i+1;
    CHKERRQ(MatSetValues(C,1,&i,3,midx,v,INSERT_VALUES));
  }
  i    = 0; midx[0] = 0; midx[1] = 1;
  v[0] = 2.0; v[1] = -1.;
  CHKERRQ(MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES));
  i    = n-1; midx[0] = n-2; midx[1] = n-1;
  v[0] = -1.0; v[1] = 2.;
  CHKERRQ(MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatMult(C,u,b));

  for (i=0; i<n; i++) {
    CHKERRQ(MatSOR(C,b,omega,SOR_FORWARD_SWEEP,0.0,1,1,x));
    CHKERRQ(VecWAXPY(e,-1.0,x,u));
    CHKERRQ(VecNorm(e,NORM_2,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"2-norm of error %g\n",(double)norm));
  }
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&e));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
