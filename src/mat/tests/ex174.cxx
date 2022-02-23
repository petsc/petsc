
static char help[] = "Tests MatConvert(), MatLoad() for MATELEMENTAL interface.\n\n";
/*
 Example:
   mpiexec -n <np> ./ex173 -fA <A_data> -fB <B_data> -orig_mat_type <type> -orig_mat_type <mat_type>
*/

#include <petscmat.h>
#include <petscmatelemental.h>

int main(int argc,char **args)
{
  Mat            A,Ae,B,Be;
  PetscErrorCode ierr;
  PetscViewer    view;
  char           file[2][PETSC_MAX_PATH_LEN];
  PetscBool      flg,flgB,isElemental,isDense,isAij,isSbaij;
  PetscScalar    one = 1.0;
  PetscMPIInt    rank,size;
  PetscInt       M,N;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Load PETSc matrices */
  ierr = PetscOptionsGetString(NULL,NULL,"-fA",file[0],sizeof(file[0]),NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"orig_");CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  PetscOptionsGetString(NULL,NULL,"-fB",file[1],sizeof(file[1]),&flgB);
  if (flgB) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&view);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(B,"orig_");CHKERRQ(ierr);
    ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatLoad(B,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  } else {
    /* Create matrix B = I */
    PetscInt rstart,rend,i;
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(B,"orig_");CHKERRQ(ierr);
    ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatSetUp(B);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      ierr = MatSetValues(B,1,&i,1,&i,&one,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = PetscObjectTypeCompare((PetscObject)A,MATELEMENTAL,&isElemental);CHKERRQ(ierr);
  if (isElemental) {
    Ae = A;
    Be = B;
    isDense = isAij = isSbaij = PETSC_FALSE;
  } else { /* Convert AIJ/DENSE/SBAIJ matrices into Elemental matrices */
    if (size == 1) {
      ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isDense);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isAij);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSbaij);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isDense);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&isAij);CHKERRQ(ierr);
       ierr = PetscObjectTypeCompare((PetscObject)A,MATMPISBAIJ,&isSbaij);CHKERRQ(ierr);
    }

    if (rank == 0) {
      if (isDense) {
        printf(" Convert DENSE matrices A and B into Elemental matrix... \n");
      } else if (isAij) {
        printf(" Convert AIJ matrices A and B into Elemental matrix... \n");
      } else if (isSbaij) {
        printf(" Convert SBAIJ matrices A and B into Elemental matrix... \n");
      } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not supported yet");
    }
    ierr = MatConvert(A, MATELEMENTAL, MAT_INITIAL_MATRIX, &Ae);CHKERRQ(ierr);
    ierr = MatConvert(B, MATELEMENTAL, MAT_INITIAL_MATRIX, &Be);CHKERRQ(ierr);

    /* Test accuracy */
    ierr = MatMultEqual(A,Ae,5,&flg);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"A != A_elemental.");
    ierr = MatMultEqual(B,Be,5,&flg);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"B != B_elemental.");
  }

  if (!isElemental) {
    ierr = MatDestroy(&Ae);CHKERRQ(ierr);
    ierr = MatDestroy(&Be);CHKERRQ(ierr);

    /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
    ierr = MatConvert(A, MATELEMENTAL, MAT_INPLACE_MATRIX, &A);CHKERRQ(ierr);
    //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: elemental

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij
      output_file: output/ex174.out

   test:
      suffix: 2
      nsize: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij
      output_file: output/ex174.out

   test:
      suffix: 2_dense
      nsize: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij -orig_mat_type dense
      output_file: output/ex174_dense.out

   test:
      suffix: 2_elemental
      nsize: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij -orig_mat_type elemental
      output_file: output/ex174_elemental.out

   test:
      suffix: 2_sbaij
      nsize: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B -orig_mat_type sbaij
      output_file: output/ex174_sbaij.out

   test:
      suffix: complex
      requires: complex double datafilespath !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/nimrod/small_112905
      output_file: output/ex174.out

   test:
      suffix: complex_2
      nsize: 4
      requires: complex double datafilespath !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/nimrod/small_112905
      output_file: output/ex174.out

   test:
      suffix: dense
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij -orig_mat_type dense

   test:
      suffix: elemental
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij -orig_mat_type elemental

   test:
      suffix: sbaij
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B -orig_mat_type sbaij

TEST*/
