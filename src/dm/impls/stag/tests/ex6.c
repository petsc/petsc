static char help[] = "Spot test DMStag->DMDA routines in 3d\n\n";
#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM              dm;
  Vec             vec;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,3,3,3,DMSTAG_STENCIL_STAR,1,NULL,NULL,NULL,&dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,10.0,0.0,10.0,0.0,10.0));

  PetscCall(DMCreateGlobalVector(dm,&vec));
  PetscCall(VecSet(vec,1.234));

  /* All element values */
  {
    DM  da;
    Vec vecda;
    PetscCall(DMStagVecSplitToDMDA(dm,vec,DMSTAG_ELEMENT,-3,&da,&vecda));
    PetscCall(DMDestroy(&da));
    PetscCall(VecDestroy(&vecda));
  }

  /* Pad element values */
  {
    DM  da;
    Vec vecda;
    PetscCall(DMStagVecSplitToDMDA(dm,vec,DMSTAG_ELEMENT,-5,&da,&vecda));
    PetscCall(DMDestroy(&da));
    PetscCall(VecDestroy(&vecda));
  }

  /* 2 element values */
  {
    DM  da;
    Vec vecda;
    PetscCall(DMStagVecSplitToDMDA(dm,vec,DMSTAG_ELEMENT,-2,&da,&vecda));
    PetscCall(DMDestroy(&da));
    PetscCall(VecDestroy(&vecda));
  }

  /* One corner value */
  {
    DM  da;
    Vec vecda;
    PetscCall(DMStagVecSplitToDMDA(dm,vec,DMSTAG_FRONT_DOWN_LEFT,2,&da,&vecda));
    PetscCall(DMDestroy(&da));
    PetscCall(VecDestroy(&vecda));
  }

  /* One edge value */
  {
    DM  da;
    Vec vecda;
    PetscCall(DMStagVecSplitToDMDA(dm,vec,DMSTAG_BACK_RIGHT,1,&da,&vecda));
    PetscCall(DMDestroy(&da));
    PetscCall(VecDestroy(&vecda));
  }

  /* One face value */
  {
    DM  da;
    Vec vecda;
    PetscCall(DMStagVecSplitToDMDA(dm,vec,DMSTAG_DOWN,0,&da,&vecda));
    PetscCall(DMDestroy(&da));
    PetscCall(VecDestroy(&vecda));
  }

  PetscCall(VecDestroy(&vec));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 12
      args: -stag_ranks_x 2 -stag_ranks_y 3 -stag_ranks_z 2

TEST*/
