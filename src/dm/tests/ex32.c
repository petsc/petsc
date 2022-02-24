static char help[] = "Tests DMDA ghost coordinates\n\n";

#include <petscdm.h>
#include <petscdmda.h>

static PetscErrorCode CompareGhostedCoords(Vec gc1,Vec gc2)
{
  PetscReal      nrm,gnrm;
  Vec            tmp;

  PetscFunctionBeginUser;
  CHKERRQ(VecDuplicate(gc1,&tmp));
  CHKERRQ(VecWAXPY(tmp,-1.0,gc1,gc2));
  CHKERRQ(VecNorm(tmp,NORM_INFINITY,&nrm));
  CHKERRMPI(MPI_Allreduce(&nrm,&gnrm,1,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"norm of difference of ghosted coordinates %8.2e\n",gnrm));
  CHKERRQ(VecDestroy(&tmp));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestQ2Q1DA(void)
{
  DM             Q2_da,Q1_da,cda;
  PetscInt       mx,my,mz;
  Vec            coords,gcoords,gcoords2;

  PetscFunctionBeginUser;
  mx   = 7;
  my   = 11;
  mz   = 13;
  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,my,mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,2,0,0,0,&Q2_da));
  CHKERRQ(DMSetFromOptions(Q2_da));
  CHKERRQ(DMSetUp(Q2_da));
  CHKERRQ(DMDASetUniformCoordinates(Q2_da,-1.0,1.0,-2.0,2.0,-3.0,3.0));
  CHKERRQ(DMGetCoordinates(Q2_da,&coords));
  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,my,mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,0,0,0,&Q1_da));
  CHKERRQ(DMSetFromOptions(Q1_da));
  CHKERRQ(DMSetUp(Q1_da));
  CHKERRQ(DMSetCoordinates(Q1_da,coords));

  /* Get ghost coordinates one way */
  CHKERRQ(DMGetCoordinatesLocal(Q1_da,&gcoords));

  /* And another */
  CHKERRQ(DMGetCoordinates(Q1_da,&coords));
  CHKERRQ(DMGetCoordinateDM(Q1_da,&cda));
  CHKERRQ(DMGetLocalVector(cda,&gcoords2));
  CHKERRQ(DMGlobalToLocalBegin(cda,coords,INSERT_VALUES,gcoords2));
  CHKERRQ(DMGlobalToLocalEnd(cda,coords,INSERT_VALUES,gcoords2));

  CHKERRQ(CompareGhostedCoords(gcoords,gcoords2));
  CHKERRQ(DMRestoreLocalVector(cda,&gcoords2));

  CHKERRQ(VecScale(coords,10.0));
  CHKERRQ(VecScale(gcoords,10.0));
  CHKERRQ(DMGetCoordinatesLocal(Q1_da,&gcoords2));
  CHKERRQ(CompareGhostedCoords(gcoords,gcoords2));

  CHKERRQ(DMDestroy(&Q2_da));
  CHKERRQ(DMDestroy(&Q1_da));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  CHKERRQ(TestQ2Q1DA());
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
