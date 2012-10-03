
static char help[] = "Demonstrates the use of fast Richardson for SOR. And\n\
also tests the MatSOR() routines.  Input parameters are:\n\
 -n <n> : problem dimension\n\n";

#include <petscksp.h>
#include <petscpc.h>

#undef __FUNCT__
#define __FUNCT__ "main"
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

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /* Create and initialize vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&b);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&ustar);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRQ(ierr);
  ierr = VecSet(ustar,1.0);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);

  /* Create and assemble matrix */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&mat);CHKERRQ(ierr);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(mat,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Compute right-hand-side vector */
  ierr = MatMult(mat,ustar,b);CHKERRQ(ierr);

  /* Create PC context and set up data structures */
  ierr = PCCreate(PETSC_COMM_WORLD,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc);CHKERRQ(ierr);
  ierr = PCSetOperators(pc,mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PCSetUp(pc);CHKERRQ(ierr);

  /* Create KSP context and set up data structures */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = PCSetOperators(pc,mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetPC(ksp,pc);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  /* Solve the problem */
  ierr = KSPGetType(ksp,&kspname);CHKERRQ(ierr);
  ierr = PCGetType(pc,&pcname);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Running %s with %s preconditioning\n",kspname,pcname);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations %D\n",its);

  /* Free data structures */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&ustar);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PCDestroy(&pc);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}



