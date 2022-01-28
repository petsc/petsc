
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
  ierr = MatCreateDense(PETSC_COMM_SELF,m,n,m,n,NULL,&Ad);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_SELF,p,q,p,q,NULL,&Bd);CHKERRQ(ierr);
  ierr = MatSetRandom(Ad,NULL);CHKERRQ(ierr);
  ierr = MatSetRandom(Bd,NULL);CHKERRQ(ierr);
  ierr = MatChop(Ad,0.2);CHKERRQ(ierr);
  ierr = MatChop(Bd,0.2);CHKERRQ(ierr);
  ierr = MatConvert(Ad,MATAIJ,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatConvert(Bd,MATAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatSeqAIJKron(A,B,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr);
  ierr = MatViewFromOptions(B,NULL,"-B_view");CHKERRQ(ierr);
  ierr = MatViewFromOptions(C,NULL,"-C_view");CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(Bd,&Bv);CHKERRQ(ierr);
  ierr = MatCreateKAIJ(A,p,q,NULL,Bv,&K);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(Bd,&Bv);CHKERRQ(ierr);
  ierr = MatMultEqual(C,K,10,&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"K*x != C*x");
  ierr = MatScale(A,1.3);CHKERRQ(ierr);
  ierr = MatScale(B,0.3);CHKERRQ(ierr);
  ierr = MatScale(Bd,0.3);CHKERRQ(ierr);
  ierr = MatSeqAIJKron(A,B,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(Bd,&Bv);CHKERRQ(ierr);
  ierr = MatKAIJSetT(K,p,q,Bv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(Bd,&Bv);CHKERRQ(ierr);
  ierr = MatMultEqual(C,K,10,&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"K*x != C*x");
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Bd);CHKERRQ(ierr);
  ierr = MatDestroy(&Ad);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      suffix: 1
      nsize: 1
      output_file: output/ex101.out

TEST*/
