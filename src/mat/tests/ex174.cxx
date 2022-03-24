
static char help[] = "Tests MatConvert(), MatLoad() for MATELEMENTAL interface.\n\n";
/*
 Example:
   mpiexec -n <np> ./ex173 -fA <A_data> -fB <B_data> -orig_mat_type <type> -orig_mat_type <mat_type>
*/

#include <petscmat.h>
#include <petscmatelemental.h>

int main(int argc,char **args)
{
  Mat         A,Ae,B,Be;
  PetscViewer view;
  char        file[2][PETSC_MAX_PATH_LEN];
  PetscBool   flg,flgB,isElemental,isDense,isAij,isSbaij;
  PetscScalar one = 1.0;
  PetscMPIInt rank,size;
  PetscInt    M,N;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Load PETSc matrices */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fA",file[0],sizeof(file[0]),NULL));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&view));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetOptionsPrefix(A,"orig_"));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,view));
  CHKERRQ(PetscViewerDestroy(&view));

  PetscOptionsGetString(NULL,NULL,"-fB",file[1],sizeof(file[1]),&flgB);
  if (flgB) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&view));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetOptionsPrefix(B,"orig_"));
    CHKERRQ(MatSetType(B,MATAIJ));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatLoad(B,view));
    CHKERRQ(PetscViewerDestroy(&view));
  } else {
    /* Create matrix B = I */
    PetscInt rstart,rend,i;
    CHKERRQ(MatGetSize(A,&M,&N));
    CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));

    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetOptionsPrefix(B,"orig_"));
    CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,N));
    CHKERRQ(MatSetType(B,MATAIJ));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatSetUp(B));
    for (i=rstart; i<rend; i++) {
      CHKERRQ(MatSetValues(B,1,&i,1,&i,&one,ADD_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }

  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATELEMENTAL,&isElemental));
  if (isElemental) {
    Ae = A;
    Be = B;
    isDense = isAij = isSbaij = PETSC_FALSE;
  } else { /* Convert AIJ/DENSE/SBAIJ matrices into Elemental matrices */
    if (size == 1) {
      CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isDense));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isAij));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSbaij));
    } else {
      CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isDense));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&isAij));
       CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATMPISBAIJ,&isSbaij));
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
    CHKERRQ(MatConvert(A, MATELEMENTAL, MAT_INITIAL_MATRIX, &Ae));
    CHKERRQ(MatConvert(B, MATELEMENTAL, MAT_INITIAL_MATRIX, &Be));

    /* Test accuracy */
    CHKERRQ(MatMultEqual(A,Ae,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"A != A_elemental.");
    CHKERRQ(MatMultEqual(B,Be,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"B != B_elemental.");
  }

  if (!isElemental) {
    CHKERRQ(MatDestroy(&Ae));
    CHKERRQ(MatDestroy(&Be));

    /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
    CHKERRQ(MatConvert(A, MATELEMENTAL, MAT_INPLACE_MATRIX, &A));
    //CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFinalize());
  return 0;
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
