
static char help[] = "Tests MatSeqAIJKron.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat               A,B,C,K,Ad,Bd;
  const PetscScalar *Bv;
  PetscInt          n = 10, m = 20, p = 7, q = 17;
  PetscBool         flg;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(MatCreateDense(PETSC_COMM_SELF,m,n,m,n,NULL,&Ad));
  CHKERRQ(MatCreateDense(PETSC_COMM_SELF,p,q,p,q,NULL,&Bd));
  CHKERRQ(MatSetRandom(Ad,NULL));
  CHKERRQ(MatSetRandom(Bd,NULL));
  CHKERRQ(MatChop(Ad,0.2));
  CHKERRQ(MatChop(Bd,0.2));
  CHKERRQ(MatConvert(Ad,MATAIJ,MAT_INITIAL_MATRIX,&A));
  CHKERRQ(MatConvert(Bd,MATAIJ,MAT_INITIAL_MATRIX,&B));
  CHKERRQ(MatSeqAIJKron(A,B,MAT_INITIAL_MATRIX,&C));
  CHKERRQ(MatViewFromOptions(A,NULL,"-A_view"));
  CHKERRQ(MatViewFromOptions(B,NULL,"-B_view"));
  CHKERRQ(MatViewFromOptions(C,NULL,"-C_view"));
  CHKERRQ(MatDenseGetArrayRead(Bd,&Bv));
  CHKERRQ(MatCreateKAIJ(A,p,q,NULL,Bv,&K));
  CHKERRQ(MatDenseRestoreArrayRead(Bd,&Bv));
  CHKERRQ(MatMultEqual(C,K,10,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"K*x != C*x");
  CHKERRQ(MatScale(A,1.3));
  CHKERRQ(MatScale(B,0.3));
  CHKERRQ(MatScale(Bd,0.3));
  CHKERRQ(MatSeqAIJKron(A,B,MAT_REUSE_MATRIX,&C));
  CHKERRQ(MatDenseGetArrayRead(Bd,&Bv));
  CHKERRQ(MatKAIJSetT(K,p,q,Bv));
  CHKERRQ(MatDenseRestoreArrayRead(Bd,&Bv));
  CHKERRQ(MatMultEqual(C,K,10,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"K*x != C*x");
  CHKERRQ(MatDestroy(&K));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&Bd));
  CHKERRQ(MatDestroy(&Ad));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      suffix: 1
      nsize: 1
      output_file: output/ex101.out

TEST*/
