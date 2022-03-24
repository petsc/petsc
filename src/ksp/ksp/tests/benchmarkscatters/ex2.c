
static char help[] = "Tests shared memory subcommunicators\n\n";
#include <petscsys.h>
#include <petscvec.h>

/*
   One can use petscmpiexec -n 3 -hosts localhost,Barrys-MacBook-Pro.local ./ex2 -info to mimic
  having two nodes that do not share common memory
*/

int main(int argc,char **args)
{
  PetscCommShared scomm;
  MPI_Comm        comm;
  PetscMPIInt     lrank,rank,size,i;
  Vec             x,y;
  VecScatter      vscat;
  IS              isstride,isblock;
  PetscViewer     singleton;
  PetscInt        indices[] = {0,1,2};

  PetscInitialize(&argc,&args,(char*)0,help);
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 3,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This example only works for 3 processes");

  CHKERRQ(PetscCommDuplicate(PETSC_COMM_WORLD,&comm,NULL));
  CHKERRQ(PetscCommSharedGet(comm,&scomm));

  for (i=0; i<size; i++) {
    CHKERRQ(PetscCommSharedGlobalToLocal(scomm,i,&lrank));
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Global rank %d shared memory comm rank %d\n",rank,i,lrank));
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));
  CHKERRQ(PetscCommDestroy(&comm));

  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,2,PETSC_DETERMINE,&x));
  CHKERRQ(VecSetBlockSize(x,2));
  CHKERRQ(VecSetValue(x,2*rank,(PetscScalar)(2*rank+10),INSERT_VALUES));
  CHKERRQ(VecSetValue(x,2*rank+1,(PetscScalar)(2*rank+1+10),INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,6,&y));
  CHKERRQ(VecSetBlockSize(y,2));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,6,0,1,&isstride));
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,2,3,indices,PETSC_COPY_VALUES,&isblock));
  CHKERRQ(VecScatterCreate(x,isblock,y,isstride,&vscat));
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&vscat));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&singleton));
  CHKERRQ(VecView(y,singleton));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&singleton));

  CHKERRQ(ISDestroy(&isstride));
  CHKERRQ(ISDestroy(&isblock));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  PetscFinalize();
  return 0;
}
