static char help[] = "Test DMStagRestrictSimple()\n\n";

#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM        dm,dm_coarse;
  Vec       vec,vec_coarse,vec_local,vec_local_coarse;
  PetscInt  dim,size_coarse;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  dim = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  switch (dim) {
    case 1:
      PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,2,3,DMSTAG_STENCIL_BOX,1,NULL,&dm));
      break;
    case 2:
      PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,8,16,PETSC_DECIDE,PETSC_DECIDE,2,3,4,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
      break;
    case 3:
      PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,8,12,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,3,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
      break;
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented!");
  }
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCoarsen(dm,MPI_COMM_NULL,&dm_coarse));

  PetscCall(DMCreateGlobalVector(dm,&vec));
  PetscCall(VecSet(vec,1.0));
  PetscCall(DMCreateLocalVector(dm,&vec_local));
  PetscCall(DMGlobalToLocal(dm,vec,INSERT_VALUES,vec_local));

  PetscCall(DMCreateGlobalVector(dm_coarse,&vec_coarse));
  PetscCall(DMCreateLocalVector(dm_coarse,&vec_local_coarse));

  PetscCall(DMStagRestrictSimple(dm,vec_local,dm_coarse,vec_local_coarse));

  PetscCall(DMLocalToGlobal(dm_coarse,vec_local_coarse,INSERT_VALUES,vec_coarse));

  PetscCall(VecGetSize(vec_coarse,&size_coarse));
  PetscCall(VecNorm(vec_coarse,NORM_1,&norm));
  PetscCheck((norm - size_coarse)/((PetscReal) size_coarse) <= PETSC_MACHINE_EPSILON * 10.0,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Numerical test failed");
  PetscCall(VecDestroy(&vec_coarse));
  PetscCall(VecDestroy(&vec));
  PetscCall(VecDestroy(&vec_local_coarse));
  PetscCall(VecDestroy(&vec_local));
  PetscCall(DMDestroy(&dm_coarse));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1d
      nsize: 1
      args: -dim 1

   test:
      suffix: 1d_par
      nsize: 4
      args: -dim 1

   test:
      suffix: 2d
      nsize: 1
      args: -dim 2

   test:
      suffix: 2d_par
      nsize: 2
      args: -dim 2

   test:
      suffix: 2d_par_2
      nsize: 8
      args: -dim 2
   test:
      suffix: 3d
      nsize: 1
      args: -dim 3

   test:
      suffix: 3d_par
      nsize: 2
      args: -dim 3
TEST*/
