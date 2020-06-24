
static char help[] = "Tests solving linear system with KSPFGMRES + PCSOR (omega != 1) on a matrix obtained from MatTransposeMatMult.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            A,Ad,B;
  PetscErrorCode ierr;
  PetscInt       N = 10, M = 3;
  PetscBool      no_inodes=PETSC_TRUE,flg;
  KSP            ksp;
  PC             pc;
  Vec            x,y;
  char           mtype[256];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-mtype",mtype,sizeof(mtype),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_inodes",&no_inodes,NULL);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,&Ad);CHKERRQ(ierr);
  ierr = MatSetRandom(Ad,NULL);CHKERRQ(ierr);
  ierr = MatConvert(Ad,flg ? mtype : MATAIJ,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatProductCreate(A,A,NULL,&B);CHKERRQ(ierr);
  ierr = MatProductSetType(B,MATPRODUCT_AtB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(B,"default");CHKERRQ(ierr);
  ierr = MatProductSetFill(B,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatProductSymbolic(B);CHKERRQ(ierr);
  if (no_inodes) {
    ierr = MatSetOption(B,MAT_USE_INODES,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = MatProductNumeric(B);CHKERRQ(ierr);
  ierr = MatTransposeMatMultEqual(A,A,B,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Wrong MatTransposeMat");CHKERRQ(ierr);
  }
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,B,B);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCSOR);CHKERRQ(ierr);
  ierr = PCSORSetOmega(pc,1.1);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPView(ksp,NULL);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&y,&x);CHKERRQ(ierr);
  ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
  ierr = PCApply(pc,x,y);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&Ad);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      nsize: 1
      suffix: 1

    test:
      nsize: 1
      suffix: 1_mpiaij
      args: -mtype mpiaij

    test:
      nsize: 3
      suffix: 2

TEST*/
