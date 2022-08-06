static char help[] = "Test MatProductReplaceMats() \n\
Modified from the code contributed by Pierre Jolivet \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscInt    n = 2,convert;
  Mat         A,B,Bdense,Conjugate;
  PetscBool   conjugate = PETSC_FALSE,equal,flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,NULL,help));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetType(A,MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqDenseSetPreallocation(A,NULL));
  PetscCall(MatMPIDenseSetPreallocation(A,NULL));
  PetscCall(MatSetRandom(A,NULL));
  PetscCall(MatViewFromOptions(A,NULL,"-A_view"));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-conjugate",&conjugate,NULL));

  for (convert = 0; convert<2; convert++) {
    /* convert dense matrix A to aij format */
    if (convert) PetscCall(MatConvert(A,MATAIJ,MAT_INPLACE_MATRIX,&A));

    /* compute B = A^T * A or  B = A^H * A */
    PetscCall(MatProductCreate(A,A,NULL,&B));

    flg = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-atb",&flg,NULL));
    if (flg) {
      PetscCall(MatProductSetType(B,MATPRODUCT_AtB));
    } else {
      PetscCall(PetscOptionsGetBool(NULL,NULL,"-ptap",&flg,NULL));
      if (flg) {
        PetscCall(MatProductSetType(B,MATPRODUCT_PtAP));
      } else {
        PetscCall(PetscOptionsGetBool(NULL,NULL,"-abt",&flg,NULL));
        if (flg) {
          PetscCall(MatProductSetType(B,MATPRODUCT_ABt));
        } else {
          PetscCall(MatProductSetType(B,MATPRODUCT_AB));
        }
      }
    }
    PetscCall(MatProductSetFromOptions(B));
    PetscCall(MatProductSymbolic(B));

    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Conjugate));
    if (conjugate) PetscCall(MatConjugate(Conjugate));

    /* replace input A by Conjugate */
    PetscCall(MatProductReplaceMats(Conjugate,NULL,NULL,B));

    PetscCall(MatProductNumeric(B));
    PetscCall(MatViewFromOptions(B,NULL,"-product_view"));

    PetscCall(MatDestroy(&Conjugate));
    if (!convert) {
      Bdense = B; B = NULL;
    }
  }

  /* Compare Bdense and B */
  PetscCall(MatMultEqual(Bdense,B,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Bdense != B");

  PetscCall(MatDestroy(&Bdense));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -conjugate false -atb
      output_file: output/ex258_1.out

   test:
      suffix: 2
      args: -conjugate true -atb
      output_file: output/ex258_1.out

   test:
      suffix: 3
      args: -conjugate false
      output_file: output/ex258_1.out

   test:
      suffix: 4
      args: -ptap
      output_file: output/ex258_1.out

   test:
      suffix: 5
      args: -abt
      output_file: output/ex258_1.out

   test:
      suffix: 6
      nsize: 2
      args: -conjugate false -atb
      output_file: output/ex258_1.out

   test:
      suffix: 7
      nsize: 2
      args: -conjugate true -atb
      output_file: output/ex258_1.out

   test:
      suffix: 8
      nsize: 2
      args: -conjugate false
      output_file: output/ex258_1.out

   test:
      suffix: 9
      nsize: 2
      args: -ptap
      output_file: output/ex258_1.out

   test:
      suffix: 10
      nsize: 2
      args: -abt
      output_file: output/ex258_1.out

   test:
      suffix: 11
      nsize: 2
      args: -conjugate true -atb -mat_product_algorithm backend
      TODO: bug: MatProductReplaceMats() does not change the product for this test

TEST*/
