
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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-gather_add",&gather_add,NULL);CHKERRQ(ierr);

  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);

  ierr = DMRedundantCreate(PETSC_COMM_WORLD,0,nredundant1,&dmred1);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmred1,&redundant1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,dmred1);CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,1,1,NULL,&da1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da1);CHKERRQ(ierr);
  ierr = DMSetUp(da1);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da1,&local1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,da1);CHKERRQ(ierr);

  ierr = DMRedundantCreate(PETSC_COMM_WORLD,1%size,nredundant2,&dmred2);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmred2,&redundant2);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,dmred2);CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,6,1,1,NULL,&da2);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da2);CHKERRQ(ierr);
  ierr = DMSetUp(da2);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da2,&local2);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,da2);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(packer,&global);CHKERRQ(ierr);
  ierr = PFCreate(PETSC_COMM_WORLD,1,1,&pf);CHKERRQ(ierr);
  ierr = PFSetType(pf,PFIDENTITY,NULL);CHKERRQ(ierr);
  ierr = PFApplyVec(pf,NULL,global);CHKERRQ(ierr);
  ierr = PFDestroy(&pf);CHKERRQ(ierr);
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMCompositeScatter(packer,global,redundant1,local1,redundant2,local2);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of redundant1 vector\n",rank);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = VecView(redundant1,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of da1 vector\n",rank);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = VecView(local1,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of redundant2 vector\n",rank);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = VecView(redundant2,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of da2 vector\n",rank);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = VecView(local2,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecGetArray(redundant1,&redundant1a);CHKERRQ(ierr);
  ierr = VecGetArray(redundant2,&redundant2a);CHKERRQ(ierr);
  for (i=0; i<nredundant1; i++) redundant1a[i] = (rank+2)*i;
  for (i=0; i<nredundant2; i++) redundant2a[i] = (rank+10)*i;
  ierr = VecRestoreArray(redundant1,&redundant1a);CHKERRQ(ierr);
  ierr = VecRestoreArray(redundant2,&redundant2a);CHKERRQ(ierr);

  ierr = DMCompositeGather(packer,gather_add ? ADD_VALUES : INSERT_VALUES,global,redundant1,local1,redundant2,local2);CHKERRQ(ierr);
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* get the global numbering for each subvector element */
  ierr = DMCompositeGetISLocalToGlobalMappings(packer,&ltog);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of redundant1 vector\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of local1 vector\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of redundant2 vector\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[2],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of local2 vector\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[3],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  for (i=0; i<4; i++) {ierr = ISLocalToGlobalMappingDestroy(&ltog[i]);CHKERRQ(ierr);}
  ierr = PetscFree(ltog);CHKERRQ(ierr);

  ierr = DMDestroy(&da1);CHKERRQ(ierr);
  ierr = DMDestroy(&dmred1);CHKERRQ(ierr);
  ierr = DMDestroy(&dmred2);CHKERRQ(ierr);
  ierr = DMDestroy(&da2);CHKERRQ(ierr);
  ierr = VecDestroy(&redundant1);CHKERRQ(ierr);
  ierr = VecDestroy(&redundant2);CHKERRQ(ierr);
  ierr = VecDestroy(&local1);CHKERRQ(ierr);
  ierr = VecDestroy(&local2);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&packer);CHKERRQ(ierr);
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
