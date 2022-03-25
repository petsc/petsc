static char help[] = "Tests DMDA ghost coordinates\n\n";

#include <petscdm.h>
#include <petscdmda.h>

static PetscErrorCode CompareGhostedCoords(Vec gc1,Vec gc2)
{
  PetscReal      nrm,gnrm;
  Vec            tmp;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(gc1,&tmp));
  PetscCall(VecWAXPY(tmp,-1.0,gc1,gc2));
  PetscCall(VecNorm(tmp,NORM_INFINITY,&nrm));
  PetscCallMPI(MPI_Allreduce(&nrm,&gnrm,1,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"norm of difference of ghosted coordinates %8.2e\n",gnrm));
  PetscCall(VecDestroy(&tmp));
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
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,my,mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,2,0,0,0,&Q2_da));
  PetscCall(DMSetFromOptions(Q2_da));
  PetscCall(DMSetUp(Q2_da));
  PetscCall(DMDASetUniformCoordinates(Q2_da,-1.0,1.0,-2.0,2.0,-3.0,3.0));
  PetscCall(DMGetCoordinates(Q2_da,&coords));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,my,mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,0,0,0,&Q1_da));
  PetscCall(DMSetFromOptions(Q1_da));
  PetscCall(DMSetUp(Q1_da));
  PetscCall(DMSetCoordinates(Q1_da,coords));

  /* Get ghost coordinates one way */
  PetscCall(DMGetCoordinatesLocal(Q1_da,&gcoords));

  /* And another */
  PetscCall(DMGetCoordinates(Q1_da,&coords));
  PetscCall(DMGetCoordinateDM(Q1_da,&cda));
  PetscCall(DMGetLocalVector(cda,&gcoords2));
  PetscCall(DMGlobalToLocalBegin(cda,coords,INSERT_VALUES,gcoords2));
  PetscCall(DMGlobalToLocalEnd(cda,coords,INSERT_VALUES,gcoords2));

  PetscCall(CompareGhostedCoords(gcoords,gcoords2));
  PetscCall(DMRestoreLocalVector(cda,&gcoords2));

  PetscCall(VecScale(coords,10.0));
  PetscCall(VecScale(gcoords,10.0));
  PetscCall(DMGetCoordinatesLocal(Q1_da,&gcoords2));
  PetscCall(CompareGhostedCoords(gcoords,gcoords2));

  PetscCall(DMDestroy(&Q2_da));
  PetscCall(DMDestroy(&Q1_da));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(TestQ2Q1DA());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
