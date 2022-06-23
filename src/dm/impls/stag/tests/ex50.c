static char help[] = "Test DMStagVecSplitToDMDA()\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM                    dm;
  Vec                   x;
  PetscBool             coords;
  PetscInt              dim,dof[4];
  PetscInt              n_loc[4];
  DMStagStencilLocation loc[4][3];

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  dim = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  coords = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-coords",&coords,NULL));

  // Create DMStag and setup set of locations to test
  switch (dim) {
    case 1:
      PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,2,DMSTAG_STENCIL_BOX,1,NULL,&dm));
      n_loc[0] = 1; loc[0][0] = DMSTAG_LEFT;
      n_loc[1] = 1; loc[1][0] = DMSTAG_ELEMENT;
      n_loc[2] = 0;
      n_loc[3] = 0;
      break;
    case 2:
      PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,PETSC_DECIDE,PETSC_DECIDE,2,1,3,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
      n_loc[0] = 1; loc[0][0] = DMSTAG_DOWN_LEFT;
      n_loc[1] = 2; loc[1][0] = DMSTAG_LEFT; loc[1][1] = DMSTAG_DOWN;
      n_loc[2] = 1; loc[2][0] = DMSTAG_ELEMENT;
      n_loc[3] = 0;
      break;
    case 3:
      PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,3,2,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,1,2,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
      n_loc[0] = 1; loc[0][0] = DMSTAG_BACK_DOWN_LEFT;
      n_loc[1] = 3; loc[1][0] = DMSTAG_DOWN_LEFT; loc[1][1] = DMSTAG_BACK_LEFT; loc[1][2] = DMSTAG_BACK_DOWN;
      n_loc[2] = 3; loc[2][0] = DMSTAG_LEFT; loc[2][1] = DMSTAG_DOWN; loc[2][2] = DMSTAG_BACK;
      n_loc[3] = 1; loc[3][0] = DMSTAG_ELEMENT;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %" PetscInt_FMT,dim);
  }

  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  if (coords) PetscCall(DMStagSetUniformCoordinatesProduct(dm,-1.0,1.0,-2.0,2.0,-3.0,3.0));

  PetscCall(DMCreateGlobalVector(dm,&x));
  PetscCall(VecSet(x,1.2345));

  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));
  for (PetscInt stratum = 0; stratum<dim+1; ++stratum) {
    for (PetscInt i_loc = 0; i_loc<n_loc[stratum]; ++i_loc){

      // Extract 3 components, padding or truncating
      {
        DM  da;
        Vec x_da;

        PetscCall(DMStagVecSplitToDMDA(dm,x,loc[stratum][i_loc],-3,&da,&x_da));
        PetscCall(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(VecView(x_da,PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(DMDestroy(&da));
        PetscCall(VecDestroy(&x_da));
      }

      // Extract individual components
      for (PetscInt c=0; c<dof[stratum]; ++c) {
        DM  da;
        Vec x_da;

        PetscCall(DMStagVecSplitToDMDA(dm,x,loc[stratum][i_loc],c,&da,&x_da));
        PetscCall(DMView(da,PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(VecView(x_da,PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(DMDestroy(&da));
        PetscCall(VecDestroy(&x_da));
      }
    }
  }

  PetscCall(VecDestroy(&x));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: !complex
      args: -dim {{1 2 3}separate output} -coords {{true false}separate output}

TEST*/
