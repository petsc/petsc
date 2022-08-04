
static char help[] = "Tests DMCreateMatrix for DMComposite.\n\n";

#include <petscdmredundant.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>
#include <petscpf.h>

int main(int argc,char **argv)
{
  ISLocalToGlobalMapping *ltog,ltogs;
  PetscMPIInt            size;
  DM                     packer;
  DM                     da,dmred;
  Mat                    M;
  PetscInt               i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(DMCompositeCreate(PETSC_COMM_WORLD,&packer));

  PetscCall(DMRedundantCreate(PETSC_COMM_WORLD,0,5,&dmred));
  PetscCall(DMCompositeAddDM(packer,dmred));
  PetscCall(DMGetLocalToGlobalMapping(dmred,&ltogs));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of dmred\n"));
  PetscCall(ISLocalToGlobalMappingView(ltogs,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&dmred));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_MIRROR,DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCompositeAddDM(packer,da));
  PetscCall(DMGetLocalToGlobalMapping(da,&ltogs));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of da\n"));
  PetscCall(ISLocalToGlobalMappingView(ltogs,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  PetscCall(DMSetMatType(packer,MATNEST));
  PetscCall(DMSetFromOptions(packer));
  PetscCall(DMCreateMatrix(packer,&M));
  PetscCall(MatView(M,NULL));
  PetscCall(MatDestroy(&M));

  /* get the global numbering for each subvector element */
  PetscCall(DMCompositeGetISLocalToGlobalMappings(packer,&ltog));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of dmred vector\n"));
  PetscCall(ISLocalToGlobalMappingView(ltog[0],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of da vector\n"));
  PetscCall(ISLocalToGlobalMappingView(ltog[1],PETSC_VIEWER_STDOUT_WORLD));
  for (i=0; i<2; i++) PetscCall(ISLocalToGlobalMappingDestroy(&ltog[i]));

  PetscCall(PetscFree(ltog));
  PetscCall(DMDestroy(&packer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: composite_nest_l2g
     nsize: {{1 2}separate output}

TEST*/
