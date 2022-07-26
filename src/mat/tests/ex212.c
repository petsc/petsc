static char help[] = "Test MatCreateSubMatrix with -mat_type nest and block sizes.\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat                    A, B, C, mats[6];
  IS                     rows[2];
  ISLocalToGlobalMapping cmap, rmap;
  const PetscInt         indices[3] = {0, 1, 2};
  PetscInt               i;
  PetscMPIInt            size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only coded for one process");
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,1));
  PetscCall(MatSetBlockSizes(A,2,1));
  PetscCall(MatSetType(A,MATAIJ));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,2,1,indices,PETSC_COPY_VALUES,&rmap));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,1,indices,PETSC_COPY_VALUES,&cmap));
  PetscCall(MatSetLocalToGlobalMapping(A,rmap,cmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&rmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&cmap));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,2,indices,PETSC_COPY_VALUES,&rmap));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,3,indices,PETSC_COPY_VALUES,&cmap));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,2,3));
  PetscCall(MatSetBlockSizes(B,1,1));
  PetscCall(MatSetType(B,MATAIJ));
  PetscCall(MatSetLocalToGlobalMapping(B,rmap,cmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&rmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&cmap));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetUp(B));
  mats[0] = A;
  mats[1] = B;
  mats[2] = A;
  mats[3] = NULL;
  mats[4] = B;
  mats[5] = A;
  PetscCall(MatCreateNest(PETSC_COMM_WORLD,2,NULL,3,NULL,mats,&C));
  PetscCall(MatSetUp(C));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(C,NULL));
  PetscCall(MatNestGetISs(C,rows,NULL));
  for (i=0; i<2; i++) {
    Mat      submat;
    IS       cols[3];
    PetscInt j;
    PetscCall(ISView(rows[i],NULL));
    PetscCall(MatCreateSubMatrix(C,rows[i],NULL,MAT_INITIAL_MATRIX,&submat));
    PetscCall(MatView(submat,NULL));
    PetscCall(MatNestGetISs(submat,NULL,cols));
    for (j=0; j<3; j++) {
      PetscCall(ISView(cols[j],NULL));
    }
    PetscCall(MatDestroy(&submat));
  }
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:

TEST*/
