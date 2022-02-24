
static char help[] = "Tests DMComposite routines.\n\n";

#include <petscdmredundant.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>
#include <petscpf.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               nredundant1 = 5,nredundant2 = 2,i;
  ISLocalToGlobalMapping *ltog;
  PetscMPIInt            rank,size;
  DM                     packer;
  Vec                    global,local1,local2,redundant1,redundant2;
  PF                     pf;
  DM                     da1,da2,dmred1,dmred2;
  PetscScalar            *redundant1a,*redundant2a;
  PetscViewer            sviewer;
  PetscBool              gather_add = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-gather_add",&gather_add,NULL));

  CHKERRQ(DMCompositeCreate(PETSC_COMM_WORLD,&packer));

  CHKERRQ(DMRedundantCreate(PETSC_COMM_WORLD,0,nredundant1,&dmred1));
  CHKERRQ(DMCreateLocalVector(dmred1,&redundant1));
  CHKERRQ(DMCompositeAddDM(packer,dmred1));

  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,1,1,NULL,&da1));
  CHKERRQ(DMSetFromOptions(da1));
  CHKERRQ(DMSetUp(da1));
  CHKERRQ(DMCreateLocalVector(da1,&local1));
  CHKERRQ(DMCompositeAddDM(packer,da1));

  CHKERRQ(DMRedundantCreate(PETSC_COMM_WORLD,1%size,nredundant2,&dmred2));
  CHKERRQ(DMCreateLocalVector(dmred2,&redundant2));
  CHKERRQ(DMCompositeAddDM(packer,dmred2));

  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,6,1,1,NULL,&da2));
  CHKERRQ(DMSetFromOptions(da2));
  CHKERRQ(DMSetUp(da2));
  CHKERRQ(DMCreateLocalVector(da2,&local2));
  CHKERRQ(DMCompositeAddDM(packer,da2));

  CHKERRQ(DMCreateGlobalVector(packer,&global));
  CHKERRQ(PFCreate(PETSC_COMM_WORLD,1,1,&pf));
  CHKERRQ(PFSetType(pf,PFIDENTITY,NULL));
  CHKERRQ(PFApplyVec(pf,NULL,global));
  CHKERRQ(PFDestroy(&pf));
  CHKERRQ(VecView(global,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(DMCompositeScatter(packer,global,redundant1,local1,redundant2,local2));
  CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of redundant1 vector\n",rank));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(VecView(redundant1,sviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of da1 vector\n",rank));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(VecView(local1,sviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of redundant2 vector\n",rank));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(VecView(redundant2,sviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of da2 vector\n",rank));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(VecView(local2,sviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecGetArray(redundant1,&redundant1a));
  CHKERRQ(VecGetArray(redundant2,&redundant2a));
  for (i=0; i<nredundant1; i++) redundant1a[i] = (rank+2)*i;
  for (i=0; i<nredundant2; i++) redundant2a[i] = (rank+10)*i;
  CHKERRQ(VecRestoreArray(redundant1,&redundant1a));
  CHKERRQ(VecRestoreArray(redundant2,&redundant2a));

  CHKERRQ(DMCompositeGather(packer,gather_add ? ADD_VALUES : INSERT_VALUES,global,redundant1,local1,redundant2,local2));
  CHKERRQ(VecView(global,PETSC_VIEWER_STDOUT_WORLD));

  /* get the global numbering for each subvector element */
  CHKERRQ(DMCompositeGetISLocalToGlobalMappings(packer,&ltog));

  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of redundant1 vector\n"));
  CHKERRQ(ISLocalToGlobalMappingView(ltog[0],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of local1 vector\n"));
  CHKERRQ(ISLocalToGlobalMappingView(ltog[1],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of redundant2 vector\n"));
  CHKERRQ(ISLocalToGlobalMappingView(ltog[2],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of local2 vector\n"));
  CHKERRQ(ISLocalToGlobalMappingView(ltog[3],PETSC_VIEWER_STDOUT_WORLD));

  for (i=0; i<4; i++) CHKERRQ(ISLocalToGlobalMappingDestroy(&ltog[i]));
  CHKERRQ(PetscFree(ltog));

  CHKERRQ(DMDestroy(&da1));
  CHKERRQ(DMDestroy(&dmred1));
  CHKERRQ(DMDestroy(&dmred2));
  CHKERRQ(DMDestroy(&da2));
  CHKERRQ(VecDestroy(&redundant1));
  CHKERRQ(VecDestroy(&redundant2));
  CHKERRQ(VecDestroy(&local1));
  CHKERRQ(VecDestroy(&local2));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&packer));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: 3

   test:
      suffix: 2
      nsize: 3
      args: -gather_add

TEST*/
