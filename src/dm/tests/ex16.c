
static char help[] = "Tests DMComposite routines.\n\n";

#include <petscdmredundant.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>
#include <petscpf.h>

int main(int argc,char **argv)
{
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-gather_add",&gather_add,NULL));

  PetscCall(DMCompositeCreate(PETSC_COMM_WORLD,&packer));

  PetscCall(DMRedundantCreate(PETSC_COMM_WORLD,0,nredundant1,&dmred1));
  PetscCall(DMCreateLocalVector(dmred1,&redundant1));
  PetscCall(DMCompositeAddDM(packer,dmred1));

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,1,1,NULL,&da1));
  PetscCall(DMSetFromOptions(da1));
  PetscCall(DMSetUp(da1));
  PetscCall(DMCreateLocalVector(da1,&local1));
  PetscCall(DMCompositeAddDM(packer,da1));

  PetscCall(DMRedundantCreate(PETSC_COMM_WORLD,1%size,nredundant2,&dmred2));
  PetscCall(DMCreateLocalVector(dmred2,&redundant2));
  PetscCall(DMCompositeAddDM(packer,dmred2));

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,6,1,1,NULL,&da2));
  PetscCall(DMSetFromOptions(da2));
  PetscCall(DMSetUp(da2));
  PetscCall(DMCreateLocalVector(da2,&local2));
  PetscCall(DMCompositeAddDM(packer,da2));

  PetscCall(DMCreateGlobalVector(packer,&global));
  PetscCall(PFCreate(PETSC_COMM_WORLD,1,1,&pf));
  PetscCall(PFSetType(pf,PFIDENTITY,NULL));
  PetscCall(PFApplyVec(pf,NULL,global));
  PetscCall(PFDestroy(&pf));
  PetscCall(VecView(global,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMCompositeScatter(packer,global,redundant1,local1,redundant2,local2));
  PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of redundant1 vector\n",rank));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(VecView(redundant1,sviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of da1 vector\n",rank));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(VecView(local1,sviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of redundant2 vector\n",rank));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(VecView(redundant2,sviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of da2 vector\n",rank));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(VecView(local2,sviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecGetArray(redundant1,&redundant1a));
  PetscCall(VecGetArray(redundant2,&redundant2a));
  for (i=0; i<nredundant1; i++) redundant1a[i] = (rank+2)*i;
  for (i=0; i<nredundant2; i++) redundant2a[i] = (rank+10)*i;
  PetscCall(VecRestoreArray(redundant1,&redundant1a));
  PetscCall(VecRestoreArray(redundant2,&redundant2a));

  PetscCall(DMCompositeGather(packer,gather_add ? ADD_VALUES : INSERT_VALUES,global,redundant1,local1,redundant2,local2));
  PetscCall(VecView(global,PETSC_VIEWER_STDOUT_WORLD));

  /* get the global numbering for each subvector element */
  PetscCall(DMCompositeGetISLocalToGlobalMappings(packer,&ltog));

  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of redundant1 vector\n"));
  PetscCall(ISLocalToGlobalMappingView(ltog[0],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of local1 vector\n"));
  PetscCall(ISLocalToGlobalMappingView(ltog[1],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of redundant2 vector\n"));
  PetscCall(ISLocalToGlobalMappingView(ltog[2],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of local2 vector\n"));
  PetscCall(ISLocalToGlobalMappingView(ltog[3],PETSC_VIEWER_STDOUT_WORLD));

  for (i=0; i<4; i++) PetscCall(ISLocalToGlobalMappingDestroy(&ltog[i]));
  PetscCall(PetscFree(ltog));

  PetscCall(DMDestroy(&da1));
  PetscCall(DMDestroy(&dmred1));
  PetscCall(DMDestroy(&dmred2));
  PetscCall(DMDestroy(&da2));
  PetscCall(VecDestroy(&redundant1));
  PetscCall(VecDestroy(&redundant2));
  PetscCall(VecDestroy(&local1));
  PetscCall(VecDestroy(&local2));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&packer));
  PetscCall(PetscFinalize());
  return 0;
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
