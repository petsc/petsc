
static char help[] = "Tests DMComposite routines.\n\n";

#include "petscdm.h"
#include "petscpf.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       nredundant1 = 5,nredundant2 = 2,i;
  ISLocalToGlobalMapping *ltog;
  PetscMPIInt    rank,size;
  PetscScalar    *redundant1,*redundant2;
  DM             packer;
  Vec            global,local1,local2;
  PF             pf;
  DM             da1,da2;
  PetscViewer    sviewer;
  PetscBool      gather_add = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(PETSC_NULL,"-gather_add",&gather_add,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);

  ierr = PetscMalloc(nredundant1*sizeof(PetscScalar),&redundant1);CHKERRQ(ierr);
  ierr = DMCompositeAddArray(packer,0,nredundant1);CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,8,1,1,PETSC_NULL,&da1);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da1,&local1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,(DM)da1);CHKERRQ(ierr);

  ierr = PetscMalloc(nredundant2*sizeof(PetscScalar),&redundant2);CHKERRQ(ierr);
  ierr = DMCompositeAddArray(packer,1%size,nredundant2);CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,6,1,1,PETSC_NULL,&da2);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da2,&local2);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,(DM)da2);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(packer,&global);CHKERRQ(ierr);
  ierr = PFCreate(PETSC_COMM_WORLD,1,1,&pf);CHKERRQ(ierr);
  ierr = PFSetType(pf,PFIDENTITY,PETSC_NULL);CHKERRQ(ierr);
  ierr = PFApplyVec(pf,PETSC_NULL,global);CHKERRQ(ierr);
  ierr = PFDestroy(pf);CHKERRQ(ierr);
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMCompositeScatter(packer,global,redundant1,local1,redundant2,local2);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of redundant1 array\n",rank);CHKERRQ(ierr);
  ierr = PetscScalarView(nredundant1,redundant1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of da1 vector\n",rank);CHKERRQ(ierr);
  ierr = PetscViewerGetSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
  ierr = VecView(local1,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of redundant2 array\n",rank);CHKERRQ(ierr);
  ierr = PetscScalarView(nredundant2,redundant2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"[%d] My part of da2 vector\n",rank);CHKERRQ(ierr);
  ierr = PetscViewerGetSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
  ierr = VecView(local2,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);

  for (i=0; i<nredundant1; i++) redundant1[i] = (rank+2)*i;
  for (i=0; i<nredundant2; i++) redundant2[i] = (rank+10)*i;

  ierr = DMCompositeGather(packer,global,gather_add?ADD_VALUES:INSERT_VALUES,redundant1,local1,redundant2,local2);CHKERRQ(ierr);
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* get the global numbering for each subvector/array element */
  ierr = DMCompositeGetISLocalToGlobalMappings(packer,&ltog);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of redundant1 array\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of local1 vector\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of redundant2 array\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[2],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local to global mapping of local2 vector\n");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog[3],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  for (i=0; i<4; i++) {ierr = ISLocalToGlobalMappingDestroy(ltog[i]);CHKERRQ(ierr);}
  ierr = PetscFree(ltog);CHKERRQ(ierr);

  ierr = DMDestroy(da1);CHKERRQ(ierr);
  ierr = DMDestroy(da2);CHKERRQ(ierr);
  ierr = VecDestroy(local1);CHKERRQ(ierr);
  ierr = VecDestroy(local2);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DMDestroy(packer);CHKERRQ(ierr);
  ierr = PetscFree(redundant1);CHKERRQ(ierr);
  ierr = PetscFree(redundant2);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
