static char help[] = "Tests DMDA ghost coordinates\n\n";

#include <petscdm.h>
#include <petscdmda.h>

static PetscErrorCode CompareGhostedCoords(Vec gc1,Vec gc2)
{
  PetscErrorCode ierr;
  PetscReal      nrm,gnrm;
  Vec            tmp;

  PetscFunctionBeginUser;
  ierr = VecDuplicate(gc1,&tmp);CHKERRQ(ierr);
  ierr = VecWAXPY(tmp,-1.0,gc1,gc2);CHKERRQ(ierr);
  ierr = VecNorm(tmp,NORM_INFINITY,&nrm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&nrm,&gnrm,1,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"norm of difference of ghosted coordinates %8.2e\n",gnrm);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestQ2Q1DA(void)
{
  DM             Q2_da,Q1_da,cda;
  PetscInt       mx,my,mz;
  Vec            coords,gcoords,gcoords2;
  PetscErrorCode ierr;

  mx   = 7;
  my   = 11;
  mz   = 13;
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,my,mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,2,0,0,0,&Q2_da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(Q2_da);CHKERRQ(ierr);
  ierr = DMSetUp(Q2_da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(Q2_da,-1.0,1.0,-2.0,2.0,-3.0,3.0);CHKERRQ(ierr);
  ierr = DMGetCoordinates(Q2_da,&coords);CHKERRQ(ierr);
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,my,mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,0,0,0,&Q1_da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(Q1_da);CHKERRQ(ierr);
  ierr = DMSetUp(Q1_da);CHKERRQ(ierr);
  ierr = DMSetCoordinates(Q1_da,coords);CHKERRQ(ierr);

  /* Get ghost coordinates one way */
  ierr = DMGetCoordinatesLocal(Q1_da,&gcoords);CHKERRQ(ierr);

  /* And another */
  ierr = DMGetCoordinates(Q1_da,&coords);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(Q1_da,&cda);CHKERRQ(ierr);
  ierr = DMGetLocalVector(cda,&gcoords2);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(cda,coords,INSERT_VALUES,gcoords2);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(cda,coords,INSERT_VALUES,gcoords2);CHKERRQ(ierr);

  ierr = CompareGhostedCoords(gcoords,gcoords2);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(cda,&gcoords2);CHKERRQ(ierr);

  ierr = VecScale(coords,10.0);CHKERRQ(ierr);
  ierr = VecScale(gcoords,10.0);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(Q1_da,&gcoords2);CHKERRQ(ierr);
  ierr = CompareGhostedCoords(gcoords,gcoords2);CHKERRQ(ierr);

  ierr = DMDestroy(&Q2_da);CHKERRQ(ierr);
  ierr = DMDestroy(&Q1_da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = TestQ2Q1DA();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
