
static char help[] = "Tests the aSA multigrid code.\n"
"Parameters:\n"
"-n n          to use a matrix size of n\n";

#include <petscdmda.h>
#include <petscksp.h>
#include <petscpcasa.h>

PetscErrorCode  Create1dLaplacian(PetscInt,Mat*);
PetscErrorCode  CalculateRhs(Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int Argc,char **Args)
{
  PetscInt        n = 60;
  PetscErrorCode  ierr;
  Mat             cmat;
  Vec             b,x;
  KSP             kspmg;
  PC              pcmg;
  DM              da;

  PetscInitialize(&Argc,&Args,(char *)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = Create1dLaplacian(n,&cmat);CHKERRQ(ierr);
  ierr = MatGetVecs(cmat,&x,0);CHKERRQ(ierr);
  ierr = MatGetVecs(cmat,&b,0);CHKERRQ(ierr);
  ierr = CalculateRhs(b);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&kspmg);CHKERRQ(ierr);
  ierr = KSPSetType(kspmg, KSPCG);CHKERRQ(ierr);

  ierr = KSPGetPC(kspmg,&pcmg);CHKERRQ(ierr);
  ierr = PCSetType(pcmg,PCASA);CHKERRQ(ierr);

  /* maybe user wants to override some of the choices */
  ierr = KSPSetFromOptions(kspmg);CHKERRQ(ierr);

  ierr = KSPSetOperators(kspmg,cmat,cmat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, n, 1, 1, 0, &da);CHKERRQ(ierr);
  ierr = DMDASetRefinementFactor(da, 3, 3, 3);CHKERRQ(ierr);
  ierr = PCSetDM(pcmg, (DM) da);CHKERRQ(ierr);

  ierr = PCASASetTolerances(pcmg, 1.e-10, 1.e-10, PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPSolve(kspmg,b,x);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspmg);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&cmat);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Create1dLaplacian"
PetscErrorCode Create1dLaplacian(PetscInt n,Mat *mat)
{
  PetscScalar    mone = -1.0,two = 2.0;
  PetscInt       i,j,loc_start,loc_end;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n,PETSC_DECIDE, PETSC_NULL, PETSC_DECIDE, PETSC_NULL, mat);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(*mat,&loc_start,&loc_end);CHKERRQ(ierr);
  for (i=loc_start; i<loc_end; i++) {
    if (i>0)   { j=i-1; ierr = MatSetValues(*mat,1,&i,1,&j,&mone,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValues(*mat,1,&i,1,&i,&two,INSERT_VALUES);CHKERRQ(ierr);
    if (i<n-1) { j=i+1; ierr = MatSetValues(*mat,1,&i,1,&j,&mone,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "CalculateRhs"
PetscErrorCode CalculateRhs(Vec u)
{
  PetscErrorCode ierr;
  PetscInt       i,n,loc_start,loc_end;
  PetscReal      h;
  PetscScalar    uu;

  PetscFunctionBegin;
  ierr = VecGetSize(u,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(u,&loc_start,&loc_end);CHKERRQ(ierr);
  h = 1.0/((PetscReal)(n+1));
  uu = 2.0*h*h;
  for (i=loc_start; i<loc_end; i++) {
    ierr = VecSetValues(u,1,&i,&uu,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
