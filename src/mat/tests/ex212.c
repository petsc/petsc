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

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Only coded for one process");
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,1));
  CHKERRQ(MatSetBlockSizes(A,2,1));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,2,1,indices,PETSC_COPY_VALUES,&rmap));
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,1,indices,PETSC_COPY_VALUES,&cmap));
  CHKERRQ(MatSetLocalToGlobalMapping(A,rmap,cmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&rmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&cmap));
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,2,indices,PETSC_COPY_VALUES,&rmap));
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,3,indices,PETSC_COPY_VALUES,&cmap));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,2,3));
  CHKERRQ(MatSetBlockSizes(B,1,1));
  CHKERRQ(MatSetType(B,MATAIJ));
  CHKERRQ(MatSetLocalToGlobalMapping(B,rmap,cmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&rmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&cmap));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetUp(B));
  mats[0] = A;
  mats[1] = B;
  mats[2] = A;
  mats[3] = NULL;
  mats[4] = B;
  mats[5] = A;
  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,2,NULL,3,NULL,mats,&C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(C,NULL));
  CHKERRQ(MatNestGetISs(C,rows,NULL));
  for (i=0; i<2; i++) {
    Mat      submat;
    IS       cols[3];
    PetscInt j;
    CHKERRQ(ISView(rows[i],NULL));
    CHKERRQ(MatCreateSubMatrix(C,rows[i],NULL,MAT_INITIAL_MATRIX,&submat));
    CHKERRQ(MatView(submat,NULL));
    CHKERRQ(MatNestGetISs(submat,NULL,cols));
    for (j=0; j<3; j++) {
      CHKERRQ(ISView(cols[j],NULL));
    }
    CHKERRQ(MatDestroy(&submat));
  }
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:

TEST*/
