
static char help[] = "Demonstrates constructing an application ordering.\n\n";

#include <petscsys.h>
#include <petscao.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n = 5;
  PetscInt       getpetsc[]  = {0,3,4},getapp[]  = {2,1,9,7}; 
  PetscInt       getpetsc1[] = {0,3,4},getapp1[] = {2,1,9,7};
  PetscInt       getpetsc2[] = {0,3,4},getapp2[] = {2,1,9,7};
  PetscMPIInt    rank,size;
  IS             ispetsc,isapp;
  AO             ao,ao1,ao2;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* create the index sets */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,rank,size,&isapp);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&ispetsc);CHKERRQ(ierr); /* natural numbering */

  /* create the application ordering */
  ierr = AOCreateBasicIS(isapp,ispetsc,&ao);CHKERRQ(ierr);
  ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = AOPetscToApplication(ao,4,getapp);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,3,getpetsc);CHKERRQ(ierr);
 
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 2,1,9,7 PetscToApplication %D %D %D %D\n",
          rank,getapp[0],getapp[1],getapp[2],getapp[3]);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 0,3,4 ApplicationToPetsc %D %D %D\n",
          rank,getpetsc[0],getpetsc[1],getpetsc[2]);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = AODestroy(ao);CHKERRQ(ierr);

  /* test MemoryScalable ao1 */
  if (!rank){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nTest AOCreateBasicMemoryScalable: \n");
  }
  ierr = AOCreateBasicMemoryScalableIS(isapp,ispetsc,&ao1);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = AOView(ao1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
 
  ierr = AOPetscToApplication(ao1,4,getapp1);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao1,3,getpetsc1);CHKERRQ(ierr);
 
  /* Check accuracy */;
  for (i=0; i<4;i++)
    if (getapp1[i] != getapp[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getapp1 %d != getapp %d",getapp1[i],getapp[i]);
  for (i=0; i<3;i++)
    if (getpetsc1[i] != getpetsc[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc1 %d != getpetsc %d",getpetsc1[i],getpetsc[i]);
  
  ierr = AODestroy(ao1);CHKERRQ(ierr);

  /* test MemoryScalable ao2: ispetsc = PETSC_NULL */
  if (!rank){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nTest AOCreateBasicMemoryScalable with ispetsc=PETSC_NULL:\n");
  }
  ierr = AOCreateBasicMemoryScalableIS(isapp,PETSC_NULL,&ao2);CHKERRQ(ierr);CHKERRQ(ierr);
 
  ierr = AOView(ao2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
 
  ierr = AOPetscToApplication(ao2,4,getapp2);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao2,3,getpetsc2);CHKERRQ(ierr);
 
  /* Check accuracy */;
  for (i=0; i<4;i++)
    if (getapp2[i] != getapp[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getapp2 %d != getapp %d",getapp2[i],getapp[i]);
  for (i=0; i<3;i++)
    if (getpetsc2[i] != getpetsc[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc2 %d != getpetsc %d",getpetsc1[i],getpetsc[i]);
  ierr = AODestroy(ao2);CHKERRQ(ierr);

  ierr = ISDestroy(ispetsc);CHKERRQ(ierr);
  ierr = ISDestroy(isapp);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
 


