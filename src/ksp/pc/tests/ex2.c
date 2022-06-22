
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /* Create and initialize vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&b));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&ustar));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&u));
  PetscCall(VecSet(ustar,1.0));
  PetscCall(VecSet(u,0.0));

  /* Create and assemble matrix */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,NULL,&mat));
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    PetscCall(MatSetValues(mat,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  PetscCall(MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  PetscCall(MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  /* Compute right-hand-side vector */
  PetscCall(MatMult(mat,ustar,b));

  /* Create PC context and set up data structures */
  PetscCall(PCCreate(PETSC_COMM_WORLD,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetOperators(pc,mat,mat));
  PetscCall(PCSetUp(pc));

  /* Create KSP context and set up data structures */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetType(ksp,KSPRICHARDSON));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PCSetOperators(pc,mat,mat));
  PetscCall(KSPSetPC(ksp,pc));
  PetscCall(KSPSetUp(ksp));

  /* Solve the problem */
  PetscCall(KSPGetType(ksp,&kspname));
  PetscCall(PCGetType(pc,&pcname));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Running %s with %s preconditioning\n",kspname,pcname));
  PetscCall(KSPSolve(ksp,b,u));
  PetscCall(VecAXPY(u,-1.0,ustar));
  PetscCall(VecNorm(u,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  if (norm > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"2 norm of error %g Number of iterations %" PetscInt_FMT "\n",(double)norm,its));
  }

  /* Free data structures */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&ustar));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&mat));
  PetscCall(PCDestroy(&pc));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_type cg -ksp_monitor_short

TEST*/
