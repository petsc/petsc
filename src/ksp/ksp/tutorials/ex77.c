#include <petsc.h>

static char help[] = "Solves a linear system with a block of right-hand sides using KSPHPDDM.\n\n";

int main(int argc,char **args)
{
  Mat               X,B;         /* computed solutions and RHS */
  Vec               cx,cb;       /* columns of X and B */
  Mat               A,KA = NULL; /* linear system matrix */
  KSP               ksp;         /* linear solver context */
  const PetscScalar *b;
  PetscScalar       *x,*S = NULL,*T = NULL;
  PetscInt          m,n,M,N = 5,i,j;
  const char        *deft = MATAIJ;
  PetscViewer       viewer;
  char              dir[PETSC_MAX_PATH_LEN],name[256],type[256];
  PetscBool         flg;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  ierr = PetscStrcpy(dir,".");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-load_dir",dir,sizeof(dir),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,sizeof(name),"%s/A_400.dat",dir);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","","");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-mat_type","Matrix type","MatSetType",MatList,deft,type,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    ierr = PetscStrcmp(type,MATKAIJ,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatConvert(A,type,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    }
    else {
      ierr = PetscCalloc2(N*N,&S,N*N,&T);CHKERRQ(ierr);
      for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
          S[i*(N+1)] = 1e+6; /* really easy problem used for testing */
          T[i*(N+1)] = 1e-2;
        }
      }
      ierr = MatCreateKAIJ(A,N,N,S,T,&KA);CHKERRQ(ierr);
    }
  }
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,NULL,&B);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,NULL,&X);CHKERRQ(ierr);
  ierr = MatSetRandom(B,NULL);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  if (!flg) {
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPHPDDM,&flg);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HPDDM)
    if (flg) {
      ierr = KSPHPDDMMatSolve(ksp,B,X);CHKERRQ(ierr);
    } else
#endif
    {
      ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
      for (n=0; n<N; n++) {
        ierr = MatDenseGetColumn(B,n,(PetscScalar**)&b);CHKERRQ(ierr);
        ierr = MatDenseGetColumn(X,n,&x);CHKERRQ(ierr);
        ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m,M,b,&cb);CHKERRQ(ierr);
        ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m,M,x,&cx);CHKERRQ(ierr);
        ierr = KSPSolve(ksp,cb,cx);CHKERRQ(ierr);
        ierr = VecDestroy(&cx);CHKERRQ(ierr);
        ierr = VecDestroy(&cb);CHKERRQ(ierr);
        ierr = MatDenseRestoreColumn(X,&x);CHKERRQ(ierr);
        ierr = MatDenseRestoreColumn(B,(PetscScalar**)&b);CHKERRQ(ierr);
      }
    }
  } else {
    ierr = KSPSetOperators(ksp,KA,KA);CHKERRQ(ierr);
    ierr = MatGetSize(KA,&M,NULL);CHKERRQ(ierr);
    /* from column- to row-major to be consistent with MatKAIJ format */
    ierr = MatTranspose(B,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatDenseGetArrayRead(B,&b);CHKERRQ(ierr);
    ierr = MatDenseGetArray(X,&x);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m*N,M,b,&cb);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m*N,M,x,&cx);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,cb,cx);CHKERRQ(ierr);
    ierr = VecDestroy(&cx);CHKERRQ(ierr);
    ierr = VecDestroy(&cb);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(B,&b);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFree2(S,T);CHKERRQ(ierr);
  ierr = MatDestroy(&KA);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: 2
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -ksp_converged_reason -ksp_max_it 500 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -mat_type {{aij sbaij}shared output}
      test:
         suffix: 1
         args:
      test:
         suffix: 2
         args: -ksp_type hpddm -pc_type asm -ksp_hpddm_type {{gmres bgmres}separate_output}
      test:
         suffix: 3
         args: -ksp_type hpddm -ksp_hpddm_recycle 10 -ksp_hpddm_type {{gcrodr bgcrodr}separate_output}

   test:
      nsize: 2
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -N 12 -ksp_converged_reason -ksp_max_it 500 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -mat_type kaij -pc_type pbjacobi -ksp_type hpddm -ksp_hpddm_type {{gmres bgmres}separate_output}

TEST*/
