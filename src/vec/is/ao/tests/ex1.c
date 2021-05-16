
static char help[] = "Demonstrates constructing an application ordering.\n\n";

#include <petscsys.h>
#include <petscao.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n = 5;
  PetscInt       getpetsc[]  = {0,3,4},getapp[]  = {2,1,9,7};
  PetscInt       getpetsc1[] = {0,3,4},getapp1[] = {2,1,9,7};
  PetscInt       getpetsc2[] = {0,3,4},getapp2[] = {2,1,9,7};
  PetscInt       getpetsc3[] = {0,3,4},getapp3[] = {2,1,9,7};
  PetscInt       getpetsc4[] = {0,3,4},getapp4[] = {2,1,9,7};
  PetscMPIInt    rank,size;
  IS             ispetsc,isapp;
  AO             ao;
  const PetscInt *app;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* create the index sets */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,rank,size,&isapp);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&ispetsc);CHKERRQ(ierr); /* natural numbering */

  /* create the application ordering */
  ierr = AOCreateBasicIS(isapp,ispetsc,&ao);CHKERRQ(ierr);
  ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = AOPetscToApplication(ao,4,getapp);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,3,getpetsc);CHKERRQ(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 2,1,9,7 PetscToApplication %D %D %D %D\n",rank,getapp[0],getapp[1],getapp[2],getapp[3]);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 0,3,4 ApplicationToPetsc %D %D %D\n",rank,getpetsc[0],getpetsc[1],getpetsc[2]);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  ierr = AODestroy(&ao);CHKERRQ(ierr);

  /* test MemoryScalable ao */
  /*-------------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTest AOCreateMemoryScalable: \n");CHKERRQ(ierr);
  ierr = AOCreateMemoryScalableIS(isapp,ispetsc,&ao);CHKERRQ(ierr);
  ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = AOPetscToApplication(ao,4,getapp1);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,3,getpetsc1);CHKERRQ(ierr);

  /* Check accuracy */;
  for (i=0; i<4; i++) {
    if (getapp1[i] != getapp[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getapp1 %d != getapp %d",getapp1[i],getapp[i]);
  }
  for (i=0; i<3; i++) {
    if (getpetsc1[i] != getpetsc[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc1 %d != getpetsc %d",getpetsc1[i],getpetsc[i]);
  }

  ierr = AODestroy(&ao);CHKERRQ(ierr);

  /* test MemoryScalable ao: ispetsc = NULL */
  /*-----------------------------------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTest AOCreateMemoryScalable with ispetsc=NULL:\n");CHKERRQ(ierr);
  ierr = AOCreateMemoryScalableIS(isapp,NULL,&ao);CHKERRQ(ierr);

  ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = AOPetscToApplication(ao,4,getapp2);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,3,getpetsc2);CHKERRQ(ierr);

  /* Check accuracy */;
  for (i=0; i<4; i++) {
    if (getapp2[i] != getapp[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getapp2 %d != getapp %d",getapp2[i],getapp[i]);
  }
  for (i=0; i<3; i++) {
    if (getpetsc2[i] != getpetsc[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc2 %d != getpetsc %d",getpetsc2[i],getpetsc[i]);
  }
  ierr = AODestroy(&ao);CHKERRQ(ierr);

  /* test AOCreateMemoryScalable() ao: */
  ierr = ISGetIndices(isapp,&app);CHKERRQ(ierr);
  ierr = AOCreateMemoryScalable(PETSC_COMM_WORLD,n,app,NULL,&ao);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isapp,&app);CHKERRQ(ierr);

  ierr = AOPetscToApplication(ao,4,getapp4);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,3,getpetsc4);CHKERRQ(ierr);

  /* Check accuracy */;
  for (i=0; i<4; i++) {
    if (getapp4[i] != getapp[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getapp4 %d != getapp %d",getapp4[i],getapp[i]);
  }
  for (i=0; i<3; i++) {
    if (getpetsc4[i] != getpetsc[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc4 %d != getpetsc %d",getpetsc4[i],getpetsc[i]);
  }
  ierr = AODestroy(&ao);CHKERRQ(ierr);

  /* test general API */
  /*------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTest general API: \n");CHKERRQ(ierr);
  ierr = AOCreate(PETSC_COMM_WORLD,&ao);CHKERRQ(ierr);
  ierr = AOSetIS(ao,isapp,ispetsc);CHKERRQ(ierr);
  ierr = AOSetType(ao,AOMEMORYSCALABLE);CHKERRQ(ierr);
  ierr = AOSetFromOptions(ao);CHKERRQ(ierr);

  /* ispetsc and isapp are nolonger used. */
  ierr = ISDestroy(&ispetsc);CHKERRQ(ierr);
  ierr = ISDestroy(&isapp);CHKERRQ(ierr);

  ierr = AOPetscToApplication(ao,4,getapp3);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,3,getpetsc3);CHKERRQ(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 2,1,9,7 PetscToApplication %D %D %D %D\n",rank,getapp3[0],getapp3[1],getapp3[2],getapp3[3]);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 0,3,4 ApplicationToPetsc %D %D %D\n",rank,getpetsc3[0],getpetsc3[1],getpetsc3[2]);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  /* Check accuracy */;
  for (i=0; i<4; i++) {
    if (getapp3[i] != getapp[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getapp3 %d != getapp %d",getapp3[i],getapp[i]);
  }
  for (i=0; i<3; i++) {
    if (getpetsc3[i] != getpetsc[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc3 %d != getpetsc %d",getpetsc3[i],getpetsc[i]);
  }

  ierr = AODestroy(&ao);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2

   test:
      suffix: 3
      nsize: 3

   test:
      suffix: 4
      nsize: 3
      args: -ao_type basic
      output_file: output/ex1_3.out

TEST*/
