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
  PetscErrorCode         ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help); if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Only coded for one process");
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,1);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(A,2,1);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,2,1,indices,PETSC_COPY_VALUES,&rmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,1,indices,PETSC_COPY_VALUES,&cmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(A,rmap,cmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&cmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,2,indices,PETSC_COPY_VALUES,&rmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,3,indices,PETSC_COPY_VALUES,&cmap);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,2,3);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(B,1,1);CHKERRQ(ierr);
  ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(B,rmap,cmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&cmap);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  mats[0] = A;
  mats[1] = B;
  mats[2] = A;
  mats[3] = NULL;
  mats[4] = B;
  mats[5] = A;
  ierr = MatCreateNest(PETSC_COMM_WORLD,2,NULL,3,NULL,mats,&C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(C,NULL);CHKERRQ(ierr);
  ierr = MatNestGetISs(C,rows,NULL);CHKERRQ(ierr);
  for (i=0; i<2; i++) {
    Mat      submat;
    IS       cols[3];
    PetscInt j;
    ierr = ISView(rows[i],NULL);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(C,rows[i],NULL,MAT_INITIAL_MATRIX,&submat);CHKERRQ(ierr);
    ierr = MatView(submat,NULL);CHKERRQ(ierr);
    ierr = MatNestGetISs(submat,NULL,cols);CHKERRQ(ierr);
    for (j=0; j<3; j++) {
      ierr = ISView(cols[j],NULL);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&submat);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:

TEST*/
