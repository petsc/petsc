
static char help[] = "VecView() with a DMDA1d vector and draw viewer.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscao.h>

PetscErrorCode apply(void *ctx,PetscInt n,const PetscScalar *x,PetscScalar *y)
{
  PetscInt i;

  for (i=0; i<n; i++) {y[3*i] = x[i]; y[3*i+1] = x[i]*x[i]; y[3*i+2] = x[i]*x[i]*x[i];}
  return 0;
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da;
  Vec            global;
  PF             pf;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,10,3,1,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(PFCreate(PETSC_COMM_WORLD,1,3,&pf));
  CHKERRQ(PFSet(pf,apply,NULL,NULL,NULL,NULL));
  CHKERRQ(PFApplyVec(pf,NULL,global));
  CHKERRQ(PFDestroy(&pf));
  CHKERRQ(VecView(global,PETSC_VIEWER_DRAW_WORLD));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      requires: x

TEST*/
