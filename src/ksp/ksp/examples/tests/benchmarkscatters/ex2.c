
static char help[] = "Tests shared memory subcommunicators\n\n";
#include <petscsys.h>
#include <petscvec.h>

/*
   One can use petscmpiexec -n 3 -hosts localhost,Barrys-MacBook-Pro.local ./ex2 -info to mimic 
  having two nodes that do not share common memory
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  PetscCommShared scomm;
  MPI_Comm        comm;
  PetscMPIInt     lrank,rank;
  Vec             x,y;
  VecScatter      vscat;
  IS              is;

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&comm,NULL);CHKERRQ(ierr);
  ierr = PetscCommSharedGet(comm,&scomm);CHKERRQ(ierr);
  ierr = PetscCommSharedGet(comm,&scomm);CHKERRQ(ierr);

  ierr = PetscCommSharedGlobalToLocal(scomm,1,&lrank);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Global rank %d shared memory comm rank %d\n",rank,1,lrank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&x);CHKERRQ(ierr);
  ierr = VecSetValue(x,rank,(PetscScalar)rank,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,3,&y);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,3,0,1,&is);CHKERRQ(ierr);
  ierr = ISToGeneral(is);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,is,y,is,&vscat);CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
