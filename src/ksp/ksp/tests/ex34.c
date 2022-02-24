
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
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-mtype",mtype,sizeof(mtype),&flg));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-no_inodes",&no_inodes,NULL));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,&Ad));
  CHKERRQ(MatSetRandom(Ad,NULL));
  CHKERRQ(MatConvert(Ad,flg ? mtype : MATAIJ,MAT_INITIAL_MATRIX,&A));
  CHKERRQ(MatProductCreate(A,A,NULL,&B));
  CHKERRQ(MatProductSetType(B,MATPRODUCT_AtB));
  CHKERRQ(MatProductSetAlgorithm(B,"default"));
  CHKERRQ(MatProductSetFill(B,PETSC_DEFAULT));
  CHKERRQ(MatProductSetFromOptions(B));
  CHKERRQ(MatProductSymbolic(B));
  if (no_inodes) {
    CHKERRQ(MatSetOption(B,MAT_USE_INODES,PETSC_FALSE));
  }
  CHKERRQ(MatProductNumeric(B));
  CHKERRQ(MatTransposeMatMultEqual(A,A,B,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Wrong MatTransposeMat"));
  }
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,B,B));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCSOR));
  CHKERRQ(PCSORSetOmega(pc,1.1));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPView(ksp,NULL));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(MatCreateVecs(B,&y,&x));
  CHKERRQ(VecSetRandom(x,NULL));
  CHKERRQ(PCApply(pc,x,y));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&Ad));
  CHKERRQ(MatDestroy(&A));
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
