
static char help[] = "Demonstrates constructing an application ordering.\n\n";

#include <petscsys.h>
#include <petscao.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* create the index sets */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,rank,size,&isapp));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&ispetsc)); /* natural numbering */

  /* create the application ordering */
  PetscCall(AOCreateBasicIS(isapp,ispetsc,&ao));
  PetscCall(AOView(ao,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(AOPetscToApplication(ao,4,getapp));
  PetscCall(AOApplicationToPetsc(ao,3,getpetsc));

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 2,1,9,7 PetscToApplication %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",rank,getapp[0],getapp[1],getapp[2],getapp[3]));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 0,3,4 ApplicationToPetsc %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",rank,getpetsc[0],getpetsc[1],getpetsc[2]));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  PetscCall(AODestroy(&ao));

  /* test MemoryScalable ao */
  /*-------------------------*/
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nTest AOCreateMemoryScalable: \n"));
  PetscCall(AOCreateMemoryScalableIS(isapp,ispetsc,&ao));
  PetscCall(AOView(ao,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(AOPetscToApplication(ao,4,getapp1));
  PetscCall(AOApplicationToPetsc(ao,3,getpetsc1));

  /* Check accuracy */;
  for (i=0; i<4; i++) {
    PetscCheckFalse(getapp1[i] != getapp[i],PETSC_COMM_SELF,PETSC_ERR_USER,"getapp1 %" PetscInt_FMT " != getapp %" PetscInt_FMT,getapp1[i],getapp[i]);
  }
  for (i=0; i<3; i++) {
    PetscCheckFalse(getpetsc1[i] != getpetsc[i],PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc1 %" PetscInt_FMT " != getpetsc %" PetscInt_FMT,getpetsc1[i],getpetsc[i]);
  }

  PetscCall(AODestroy(&ao));

  /* test MemoryScalable ao: ispetsc = NULL */
  /*-----------------------------------------------*/
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nTest AOCreateMemoryScalable with ispetsc=NULL:\n"));
  PetscCall(AOCreateMemoryScalableIS(isapp,NULL,&ao));

  PetscCall(AOView(ao,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(AOPetscToApplication(ao,4,getapp2));
  PetscCall(AOApplicationToPetsc(ao,3,getpetsc2));

  /* Check accuracy */;
  for (i=0; i<4; i++) {
    PetscCheckFalse(getapp2[i] != getapp[i],PETSC_COMM_SELF,PETSC_ERR_USER,"getapp2 %" PetscInt_FMT " != getapp %" PetscInt_FMT,getapp2[i],getapp[i]);
  }
  for (i=0; i<3; i++) {
    PetscCheckFalse(getpetsc2[i] != getpetsc[i],PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc2 %" PetscInt_FMT " != getpetsc %" PetscInt_FMT,getpetsc2[i],getpetsc[i]);
  }
  PetscCall(AODestroy(&ao));

  /* test AOCreateMemoryScalable() ao: */
  PetscCall(ISGetIndices(isapp,&app));
  PetscCall(AOCreateMemoryScalable(PETSC_COMM_WORLD,n,app,NULL,&ao));
  PetscCall(ISRestoreIndices(isapp,&app));

  PetscCall(AOPetscToApplication(ao,4,getapp4));
  PetscCall(AOApplicationToPetsc(ao,3,getpetsc4));

  /* Check accuracy */;
  for (i=0; i<4; i++) {
    PetscCheckFalse(getapp4[i] != getapp[i],PETSC_COMM_SELF,PETSC_ERR_USER,"getapp4 %" PetscInt_FMT " != getapp %" PetscInt_FMT,getapp4[i],getapp[i]);
  }
  for (i=0; i<3; i++) {
    PetscCheckFalse(getpetsc4[i] != getpetsc[i],PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc4 %" PetscInt_FMT " != getpetsc %" PetscInt_FMT,getpetsc4[i],getpetsc[i]);
  }
  PetscCall(AODestroy(&ao));

  /* test general API */
  /*------------------*/
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nTest general API: \n"));
  PetscCall(AOCreate(PETSC_COMM_WORLD,&ao));
  PetscCall(AOSetIS(ao,isapp,ispetsc));
  PetscCall(AOSetType(ao,AOMEMORYSCALABLE));
  PetscCall(AOSetFromOptions(ao));

  /* ispetsc and isapp are nolonger used. */
  PetscCall(ISDestroy(&ispetsc));
  PetscCall(ISDestroy(&isapp));

  PetscCall(AOPetscToApplication(ao,4,getapp3));
  PetscCall(AOApplicationToPetsc(ao,3,getpetsc3));

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 2,1,9,7 PetscToApplication %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",rank,getapp3[0],getapp3[1],getapp3[2],getapp3[3]));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] 0,3,4 ApplicationToPetsc %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",rank,getpetsc3[0],getpetsc3[1],getpetsc3[2]));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  /* Check accuracy */;
  for (i=0; i<4; i++) {
    PetscCheckFalse(getapp3[i] != getapp[i],PETSC_COMM_SELF,PETSC_ERR_USER,"getapp3 %" PetscInt_FMT " != getapp %" PetscInt_FMT,getapp3[i],getapp[i]);
  }
  for (i=0; i<3; i++) {
    PetscCheckFalse(getpetsc3[i] != getpetsc[i],PETSC_COMM_SELF,PETSC_ERR_USER,"getpetsc3 %" PetscInt_FMT " != getpetsc %" PetscInt_FMT,getpetsc3[i],getpetsc[i]);
  }

  PetscCall(AODestroy(&ao));
  PetscCall(PetscFinalize());
  return 0;
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
