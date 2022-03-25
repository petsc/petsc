
static char help[] = "Tests MatSeqAIJKron.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat               A,B,C,K,Ad,Bd;
  const PetscScalar *Bv;
  PetscInt          n = 10, m = 20, p = 7, q = 17;
  PetscBool         flg;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(MatCreateDense(PETSC_COMM_SELF,m,n,m,n,NULL,&Ad));
  PetscCall(MatCreateDense(PETSC_COMM_SELF,p,q,p,q,NULL,&Bd));
  PetscCall(MatSetRandom(Ad,NULL));
  PetscCall(MatSetRandom(Bd,NULL));
  PetscCall(MatChop(Ad,0.2));
  PetscCall(MatChop(Bd,0.2));
  PetscCall(MatConvert(Ad,MATAIJ,MAT_INITIAL_MATRIX,&A));
  PetscCall(MatConvert(Bd,MATAIJ,MAT_INITIAL_MATRIX,&B));
  PetscCall(MatSeqAIJKron(A,B,MAT_INITIAL_MATRIX,&C));
  PetscCall(MatViewFromOptions(A,NULL,"-A_view"));
  PetscCall(MatViewFromOptions(B,NULL,"-B_view"));
  PetscCall(MatViewFromOptions(C,NULL,"-C_view"));
  PetscCall(MatDenseGetArrayRead(Bd,&Bv));
  PetscCall(MatCreateKAIJ(A,p,q,NULL,Bv,&K));
  PetscCall(MatDenseRestoreArrayRead(Bd,&Bv));
  PetscCall(MatMultEqual(C,K,10,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"K*x != C*x");
  PetscCall(MatScale(A,1.3));
  PetscCall(MatScale(B,0.3));
  PetscCall(MatScale(Bd,0.3));
  PetscCall(MatSeqAIJKron(A,B,MAT_REUSE_MATRIX,&C));
  PetscCall(MatDenseGetArrayRead(Bd,&Bv));
  PetscCall(MatKAIJSetT(K,p,q,Bv));
  PetscCall(MatDenseRestoreArrayRead(Bd,&Bv));
  PetscCall(MatMultEqual(C,K,10,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"K*x != C*x");
  PetscCall(MatDestroy(&K));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Bd));
  PetscCall(MatDestroy(&Ad));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: 1
      nsize: 1
      output_file: output/ex101.out

TEST*/
