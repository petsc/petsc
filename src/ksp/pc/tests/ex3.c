
static char help[] = "Demonstrates the use of fast Richardson for SOR. And\n\
also tests the MatSOR() routines.  Input parameters are:\n\
 -n <n> : problem dimension\n\n";

#include <petscksp.h>
#include <petscpc.h>

int main(int argc,char **args)
{
  Mat            mat;          /* matrix */
  Vec            b,ustar,u;  /* vectors (RHS, exact solution, approx solution) */
  PC             pc;           /* PC context */
  KSP            ksp;          /* KSP context */
  PetscErrorCode ierr;
  PetscInt       n = 10,i,its,col[3];
  PetscScalar    value[3];
  KSPType        kspname;
  PCType         pcname;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* Create and initialize vectors */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&b));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&ustar));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&u));
  CHKERRQ(VecSet(ustar,1.0));
  CHKERRQ(VecSet(u,0.0));

  /* Create and assemble matrix */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&mat));
  CHKERRQ(MatSetType(mat,MATSEQAIJ));
  CHKERRQ(MatSetSizes(mat,n,n,n,n));
  CHKERRQ(MatSetFromOptions(mat));
  CHKERRQ(MatSeqAIJSetPreallocation(mat,3,NULL));
  CHKERRQ(MatSeqBAIJSetPreallocation(mat,1,3,NULL));
  CHKERRQ(MatSeqSBAIJSetPreallocation(mat,1,3,NULL));
  CHKERRQ(MatSeqSELLSetPreallocation(mat,3,NULL));
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
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations %D\n",its));

  /* Free data structures */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&ustar));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(PCDestroy(&pc));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
     args: -ksp_type gmres -ksp_monitor_short -pc_type sor -pc_sor_symmetric
     output_file: output/ex3_1.out
     test:
       suffix: sor_aij
     test:
       suffix: sor_seqbaij
       args: -mat_type seqbaij
     test:
       suffix: sor_seqsbaij
       args: -mat_type seqbaij
     test:
       suffix: sor_seqsell
       args: -mat_type seqsell

TEST*/
