
static char help[] = "Tests PC and KSP on a tridiagonal matrix.  Note that most\n\
users should employ the KSP interface instead of using PC directly.\n\n";


#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            mat;          /* matrix */
  Vec            b,ustar,u;  /* vectors (RHS, exact solution, approx solution) */
  PC             pc;           /* PC context */
  KSP            ksp;          /* KSP context */
  PetscErrorCode ierr;
  PetscInt       n = 10,i,its,col[3];
  PetscScalar    value[3];
  PCType         pcname;
  KSPType        kspname;
  PetscReal      norm,tol=1000.*PETSC_MACHINE_EPSILON;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Create and initialize vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&b);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&ustar);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRQ(ierr);
  ierr = VecSet(ustar,1.0);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);

  /* Create and assemble matrix */
  ierr     = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,NULL,&mat);CHKERRQ(ierr);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(mat,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Compute right-hand-side vector */
  ierr = MatMult(mat,ustar,b);CHKERRQ(ierr);

  /* Create PC context and set up data structures */
  ierr = PCCreate(PETSC_COMM_WORLD,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc);CHKERRQ(ierr);
  ierr = PCSetOperators(pc,mat,mat);CHKERRQ(ierr);
  ierr = PCSetUp(pc);CHKERRQ(ierr);

  /* Create KSP context and set up data structures */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = PCSetOperators(pc,mat,mat);CHKERRQ(ierr);
  ierr = KSPSetPC(ksp,pc);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  /* Solve the problem */
  ierr = KSPGetType(ksp,&kspname);CHKERRQ(ierr);
  ierr = PCGetType(pc,&pcname);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Running %s with %s preconditioning\n",kspname,pcname);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,ustar);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"2 norm of error %g Number of iterations %D\n",(double)norm,its);CHKERRQ(ierr);
  }

  /* Free data structures */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&ustar);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PCDestroy(&pc);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}





/*TEST

   test:
      args: -ksp_type cg -ksp_monitor_short

TEST*/
