static char help[] = "Spot test DMStag->DMDA routines in 3d\n\n";
#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  DM              dm;
  Vec             vec;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,3,3,3,DMSTAG_STENCIL_STAR,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,10.0,0.0,10.0,0.0,10.0);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&vec);CHKERRQ(ierr);
  ierr = VecSet(vec,1.234);CHKERRQ(ierr);

  /* All element values */
  {
    DM  da;
    Vec vecda;
    ierr = DMStagVecSplitToDMDA(dm,vec,DMSTAG_ELEMENT,-3,&da,&vecda);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
    ierr = VecDestroy(&vecda);CHKERRQ(ierr);
  }

  /* Pad element values */
  {
    DM  da;
    Vec vecda;
    ierr = DMStagVecSplitToDMDA(dm,vec,DMSTAG_ELEMENT,-5,&da,&vecda);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
    ierr = VecDestroy(&vecda);CHKERRQ(ierr);
  }

  /* 2 element values */
  {
    DM  da;
    Vec vecda;
    ierr = DMStagVecSplitToDMDA(dm,vec,DMSTAG_ELEMENT,-2,&da,&vecda);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
    ierr = VecDestroy(&vecda);CHKERRQ(ierr);
  }

  /* One corner value */
  {
    DM  da;
    Vec vecda;
    ierr = DMStagVecSplitToDMDA(dm,vec,DMSTAG_FRONT_DOWN_LEFT,2,&da,&vecda);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
    ierr = VecDestroy(&vecda);CHKERRQ(ierr);
  }

  /* One edge value */
  {
    DM  da;
    Vec vecda;
    ierr = DMStagVecSplitToDMDA(dm,vec,DMSTAG_BACK_RIGHT,1,&da,&vecda);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
    ierr = VecDestroy(&vecda);CHKERRQ(ierr);
  }

  /* One face value */
  {
    DM  da;
    Vec vecda;
    ierr = DMStagVecSplitToDMDA(dm,vec,DMSTAG_DOWN,0,&da,&vecda);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
    ierr = VecDestroy(&vecda);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 12
      args: -stag_ranks_x 2 -stag_ranks_y 3 -stag_ranks_z 2

TEST*/
