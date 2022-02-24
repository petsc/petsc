static char help[] = "Spot test DMStag->DMDA routines in 3d\n\n";
#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  DM              dm;
  Vec             vec;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,3,3,3,DMSTAG_STENCIL_STAR,1,NULL,NULL,NULL,&dm));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(DMStagSetUniformCoordinatesProduct(dm,0.0,10.0,0.0,10.0,0.0,10.0));

  CHKERRQ(DMCreateGlobalVector(dm,&vec));
  CHKERRQ(VecSet(vec,1.234));

  /* All element values */
  {
    DM  da;
    Vec vecda;
    CHKERRQ(DMStagVecSplitToDMDA(dm,vec,DMSTAG_ELEMENT,-3,&da,&vecda));
    CHKERRQ(DMDestroy(&da));
    CHKERRQ(VecDestroy(&vecda));
  }

  /* Pad element values */
  {
    DM  da;
    Vec vecda;
    CHKERRQ(DMStagVecSplitToDMDA(dm,vec,DMSTAG_ELEMENT,-5,&da,&vecda));
    CHKERRQ(DMDestroy(&da));
    CHKERRQ(VecDestroy(&vecda));
  }

  /* 2 element values */
  {
    DM  da;
    Vec vecda;
    CHKERRQ(DMStagVecSplitToDMDA(dm,vec,DMSTAG_ELEMENT,-2,&da,&vecda));
    CHKERRQ(DMDestroy(&da));
    CHKERRQ(VecDestroy(&vecda));
  }

  /* One corner value */
  {
    DM  da;
    Vec vecda;
    CHKERRQ(DMStagVecSplitToDMDA(dm,vec,DMSTAG_FRONT_DOWN_LEFT,2,&da,&vecda));
    CHKERRQ(DMDestroy(&da));
    CHKERRQ(VecDestroy(&vecda));
  }

  /* One edge value */
  {
    DM  da;
    Vec vecda;
    CHKERRQ(DMStagVecSplitToDMDA(dm,vec,DMSTAG_BACK_RIGHT,1,&da,&vecda));
    CHKERRQ(DMDestroy(&da));
    CHKERRQ(VecDestroy(&vecda));
  }

  /* One face value */
  {
    DM  da;
    Vec vecda;
    CHKERRQ(DMStagVecSplitToDMDA(dm,vec,DMSTAG_DOWN,0,&da,&vecda));
    CHKERRQ(DMDestroy(&da));
    CHKERRQ(VecDestroy(&vecda));
  }

  CHKERRQ(VecDestroy(&vec));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 12
      args: -stag_ranks_x 2 -stag_ranks_y 3 -stag_ranks_z 2

TEST*/
