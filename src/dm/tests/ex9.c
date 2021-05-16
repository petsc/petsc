
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);

  ierr = DMRedundantCreate(PETSC_COMM_WORLD,0,5,&dmred);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,dmred);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dmred,&ltogs);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of dmred\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltogs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&dmred);CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_MIRROR,DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,da);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&ltogs);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of da\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltogs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = DMSetMatType(packer,MATNEST);CHKERRQ(ierr);
  ierr = DMSetFromOptions(packer);CHKERRQ(ierr);
  ierr = DMCreateMatrix(packer,&M);CHKERRQ(ierr);
  ierr = MatView(M,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);

  /* get the global numbering for each subvector element */
  ierr = DMCompositeGetISLocalToGlobalMappings(packer,&ltog);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of dmred vector\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of da vector\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  for (i=0; i<2; i++) {ierr = ISLocalToGlobalMappingDestroy(&ltog[i]);CHKERRQ(ierr);}

  ierr = PetscFree(ltog);CHKERRQ(ierr);
  ierr = DMDestroy(&packer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     suffix: composite_nest_l2g
     nsize: {{1 2}separate output}

TEST*/
