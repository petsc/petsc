
static char help[] = "VecView() with a DMDA1d vector and draw viewer.\n\n";

#include <petscdmda.h>
#include <petscao.h>

PetscErrorCode apply(void *ctx,PetscInt n,const PetscScalar *x,PetscScalar *y)
{
  PetscInt i;

  for (i=0; i<n; i++) {y[3*i] = x[i]; y[3*i+1] = x[i]*x[i]; y[3*i+2] = x[i]*x[i]*x[i];}
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da;
  Vec            global;
  PF             pf;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,10,3,1,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = PFCreate(PETSC_COMM_WORLD,1,3,&pf);CHKERRQ(ierr);
  ierr = PFSet(pf,apply,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PFApplyVec(pf,PETSC_NULL,global);CHKERRQ(ierr);
  ierr = PFDestroy(&pf);CHKERRQ(ierr);
  ierr = VecView(global,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
