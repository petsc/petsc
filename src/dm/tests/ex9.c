
static char help[] = "Tests DMCreateMatrix for DMComposite.\n\n";

#include <petscdmredundant.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>
#include <petscpf.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  ISLocalToGlobalMapping *ltog,ltogs;
  PetscMPIInt            size;
  DM                     packer;
  DM                     da,dmred;
  Mat                    M;
  PetscInt               i;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(DMCompositeCreate(PETSC_COMM_WORLD,&packer));

  CHKERRQ(DMRedundantCreate(PETSC_COMM_WORLD,0,5,&dmred));
  CHKERRQ(DMCompositeAddDM(packer,dmred));
  CHKERRQ(DMGetLocalToGlobalMapping(dmred,&ltogs));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of dmred\n"));
  CHKERRQ(ISLocalToGlobalMappingView(ltogs,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&dmred));

  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_MIRROR,DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCompositeAddDM(packer,da));
  CHKERRQ(DMGetLocalToGlobalMapping(da,&ltogs));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of da\n"));
  CHKERRQ(ISLocalToGlobalMappingView(ltogs,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&da));

  CHKERRQ(DMSetMatType(packer,MATNEST));
  CHKERRQ(DMSetFromOptions(packer));
  CHKERRQ(DMCreateMatrix(packer,&M));
  CHKERRQ(MatView(M,NULL));
  CHKERRQ(MatDestroy(&M));

  /* get the global numbering for each subvector element */
  CHKERRQ(DMCompositeGetISLocalToGlobalMappings(packer,&ltog));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of dmred vector\n"));
  CHKERRQ(ISLocalToGlobalMappingView(ltog[0],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of da vector\n"));
  CHKERRQ(ISLocalToGlobalMappingView(ltog[1],PETSC_VIEWER_STDOUT_WORLD));
  for (i=0; i<2; i++) CHKERRQ(ISLocalToGlobalMappingDestroy(&ltog[i]));

  CHKERRQ(PetscFree(ltog));
  CHKERRQ(DMDestroy(&packer));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     suffix: composite_nest_l2g
     nsize: {{1 2}separate output}

TEST*/
