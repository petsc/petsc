
static char help[] = "Tests PC and KSP on a tridiagonal matrix.  Note that most\n\
users should employ the KSP interface instead of using PC directly.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            mat;          /* matrix */
  Vec            b,ustar,u;  /* vectors (RHS, exact solution, approx solution) */
  PC             pc;           /* PC context */
  KSP            ksp;          /* KSP context */
  PetscInt       n = 10,i,its,col[3];
  PetscScalar    value[3];
  PCType         pcname;
  KSPType        kspname;
  PetscReal      norm,tol=1000.*PETSC_MACHINE_EPSILON;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  /* Create and initialize vectors */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&b));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&ustar));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&u));
  CHKERRQ(VecSet(ustar,1.0));
  CHKERRQ(VecSet(u,0.0));

  /* Create and assemble matrix */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,NULL,&mat));
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    CHKERRQ(MatSetValues(mat,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  CHKERRQ(MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  CHKERRQ(MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  /* Compute right-hand-side vector */
  CHKERRQ(MatMult(mat,ustar,b));

  /* Create PC context and set up data structures */
  CHKERRQ(PCCreate(PETSC_COMM_WORLD,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  CHKERRQ(PCSetFromOptions(pc));
  CHKERRQ(PCSetOperators(pc,mat,mat));
  CHKERRQ(PCSetUp(pc));

  /* Create KSP context and set up data structures */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPRICHARDSON));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(PCSetOperators(pc,mat,mat));
  CHKERRQ(KSPSetPC(ksp,pc));
  CHKERRQ(KSPSetUp(ksp));

  /* Solve the problem */
  CHKERRQ(KSPGetType(ksp,&kspname));
  CHKERRQ(PCGetType(pc,&pcname));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Running %s with %s preconditioning\n",kspname,pcname));
  CHKERRQ(KSPSolve(ksp,b,u));
  CHKERRQ(VecAXPY(u,-1.0,ustar));
  CHKERRQ(VecNorm(u,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"2 norm of error %g Number of iterations %D\n",(double)norm,its));
  }

  /* Free data structures */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&ustar));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(PCDestroy(&pc));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_type cg -ksp_monitor_short

TEST*/
