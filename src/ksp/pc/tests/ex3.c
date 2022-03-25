
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
  PetscInt       n = 10,i,its,col[3];
  PetscScalar    value[3];
  KSPType        kspname;
  PCType         pcname;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* Create and initialize vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&b));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&ustar));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&u));
  PetscCall(VecSet(ustar,1.0));
  PetscCall(VecSet(u,0.0));

  /* Create and assemble matrix */
  PetscCall(MatCreate(PETSC_COMM_SELF,&mat));
  PetscCall(MatSetType(mat,MATSEQAIJ));
  PetscCall(MatSetSizes(mat,n,n,n,n));
  PetscCall(MatSetFromOptions(mat));
  PetscCall(MatSeqAIJSetPreallocation(mat,3,NULL));
  PetscCall(MatSeqBAIJSetPreallocation(mat,1,3,NULL));
  PetscCall(MatSeqSBAIJSetPreallocation(mat,1,3,NULL));
  PetscCall(MatSeqSELLSetPreallocation(mat,3,NULL));
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
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations %D\n",its));

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
