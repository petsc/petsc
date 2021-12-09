static char help[] = "Tests various routines for MATSHELL\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat B;
};

static PetscErrorCode MatGetDiagonal_User(Mat A,Vec X)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatGetDiagonal(user->B,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_User(Mat A,Vec X,Vec Y)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatMult(user->B,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_User(Mat A,Vec X,Vec Y)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->B,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_User(Mat A,Mat X,MatStructure str)
{
  User           user,userX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = MatShellGetContext(X,&userX);CHKERRQ(ierr);
  if (user != userX) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"This should not happen");
  ierr = PetscObjectReference((PetscObject)user->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_User(Mat A)
{
  User           user;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&user);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)user->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  User           user;
  Mat            A,S;
  PetscScalar    *data,diag = 1.3;
  PetscReal      tol = PETSC_SMALL;
  PetscInt       i,j,m = PETSC_DECIDE,n = PETSC_DECIDE,M = 17,N = 15,s1,s2;
  PetscInt       test, ntest = 2;
  PetscMPIInt    rank,size;
  PetscBool      nc = PETSC_FALSE, cong, flg;
  PetscBool      ronl = PETSC_TRUE;
  PetscBool      randomize = PETSC_FALSE, submat = PETSC_FALSE;
  PetscBool      keep = PETSC_FALSE;
  PetscBool      testzerorows = PETSC_TRUE, testdiagscale = PETSC_TRUE, testgetdiag = PETSC_TRUE, testsubmat = PETSC_TRUE;
  PetscBool      testshift = PETSC_TRUE, testscale = PETSC_TRUE, testdup = PETSC_TRUE, testreset = PETSC_TRUE;
  PetscBool      testaxpy = PETSC_TRUE, testaxpyd = PETSC_TRUE, testaxpyerr = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ml",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nl",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-square_nc",&nc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-rows_only",&ronl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-randomize",&randomize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-submat",&submat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_zerorows",&testzerorows,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_diagscale",&testdiagscale,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_getdiag",&testgetdiag,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_shift",&testshift,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_scale",&testscale,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_dup",&testdup,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_reset",&testreset,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_submat",&testsubmat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_axpy",&testaxpy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_axpy_different",&testaxpyd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_axpy_error",&testaxpyerr,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-loop",&ntest,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-tol",&tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-diag",&diag,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-keep",&keep,NULL);CHKERRQ(ierr);
  /* This tests square matrices with different row/col layout */
  if (nc && size > 1) {
    M = PetscMax(PetscMax(N,M),1);
    N = M;
    m = n = 0;
    if (rank == 0) { m = M-1; n = 1; }
    else if (rank == 1) { m = 1; n = N-1; }
  }
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,n,M,N,NULL,&A);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&s1,NULL);CHKERRQ(ierr);
  s2   = 1;
  while (s2 < M) s2 *= 10;
  ierr = MatDenseGetArray(A,&data);CHKERRQ(ierr);
  for (j = 0; j < N; j++) {
    for (i = 0; i < m; i++) {
      data[j*m + i] = s2*j + i + s1 + 1;
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (submat) {
    Mat      A2;
    IS       r,c;
    PetscInt rst,ren,cst,cen;

    ierr = MatGetOwnershipRange(A,&rst,&ren);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(A,&cst,&cen);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)A),(ren-rst)/2,rst,1,&r);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)A),(cen-cst)/2,cst,1,&c);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(A,r,c,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
    ierr = ISDestroy(&r);CHKERRQ(ierr);
    ierr = ISDestroy(&c);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    A = A2;
  }

  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatHasCongruentLayouts(A,&cong);CHKERRQ(ierr);

  ierr = MatConvert(A,MATAIJ,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,keep);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"initial");CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-view_mat");CHKERRQ(ierr);

  ierr = PetscNew(&user);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD,m,n,M,N,user,&S);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MatMult_User);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_User);CHKERRQ(ierr);
  if (cong) {
    ierr = MatShellSetOperation(S,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_User);CHKERRQ(ierr);
  }
  ierr = MatShellSetOperation(S,MATOP_COPY,(void (*)(void))MatCopy_User);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_DESTROY,(void (*)(void))MatDestroy_User);CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&user->B);CHKERRQ(ierr);

  /* Square and rows only scaling */
  ronl = cong ? ronl : PETSC_TRUE;

  for (test = 0; test < ntest; test++) {
    PetscReal err;

    ierr = MatMultAddEqual(A,S,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mult add\n",test);CHKERRQ(ierr);
    }
    ierr = MatMultTransposeAddEqual(A,S,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mult add (T)\n",test);CHKERRQ(ierr);
    }
    if (testzerorows) {
      Mat       ST,B,C,BT,BTT;
      IS        zr;
      Vec       x = NULL, b1 = NULL, b2 = NULL;
      PetscInt  *idxs = NULL, nr = 0;

      if (rank == (test%size)) {
        nr = 1;
        ierr = PetscMalloc1(nr,&idxs);CHKERRQ(ierr);
        if (test%2) {
          idxs[0] = (2*M - 1 - test/2)%M;
        } else {
          idxs[0] = (test/2)%M;
        }
        idxs[0] = PetscMax(idxs[0],0);
      }
      ierr = ISCreateGeneral(PETSC_COMM_WORLD,nr,idxs,PETSC_OWN_POINTER,&zr);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)zr,"ZR");CHKERRQ(ierr);
      ierr = ISViewFromOptions(zr,NULL,"-view_is");CHKERRQ(ierr);
      ierr = MatCreateVecs(A,&x,&b1);CHKERRQ(ierr);
      if (randomize) {
        ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
        ierr = VecSetRandom(b1,NULL);CHKERRQ(ierr);
      } else {
        ierr = VecSet(x,11.4);CHKERRQ(ierr);
        ierr = VecSet(b1,-14.2);CHKERRQ(ierr);
      }
      ierr = VecDuplicate(b1,&b2);CHKERRQ(ierr);
      ierr = VecCopy(b1,b2);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)b1,"A_B1");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)b2,"A_B2");CHKERRQ(ierr);
      if (size > 1 && !cong) { /* MATMPIAIJ ZeroRows and ZeroRowsColumns are buggy in this case */
        ierr = VecDestroy(&b1);CHKERRQ(ierr);
      }
      if (ronl) {
        ierr = MatZeroRowsIS(A,zr,diag,x,b1);CHKERRQ(ierr);
        ierr = MatZeroRowsIS(S,zr,diag,x,b2);CHKERRQ(ierr);
      } else {
        ierr = MatZeroRowsColumnsIS(A,zr,diag,x,b1);CHKERRQ(ierr);
        ierr = MatZeroRowsColumnsIS(S,zr,diag,x,b2);CHKERRQ(ierr);
        ierr = ISDestroy(&zr);CHKERRQ(ierr);
        /* Mix zerorows and zerorowscols */
        nr   = 0;
        idxs = NULL;
        if (rank == 0) {
          nr   = 1;
          ierr = PetscMalloc1(nr,&idxs);CHKERRQ(ierr);
          if (test%2) {
            idxs[0] = (3*M - 2 - test/2)%M;
          } else {
            idxs[0] = (test/2+1)%M;
          }
          idxs[0] = PetscMax(idxs[0],0);
        }
        ierr = ISCreateGeneral(PETSC_COMM_WORLD,nr,idxs,PETSC_OWN_POINTER,&zr);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)zr,"ZR2");CHKERRQ(ierr);
        ierr = ISViewFromOptions(zr,NULL,"-view_is");CHKERRQ(ierr);
        ierr = MatZeroRowsIS(A,zr,diag*2.0+PETSC_SMALL,NULL,NULL);CHKERRQ(ierr);
        ierr = MatZeroRowsIS(S,zr,diag*2.0+PETSC_SMALL,NULL,NULL);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&zr);CHKERRQ(ierr);

      if (b1) {
        Vec b;

        ierr = VecViewFromOptions(b1,NULL,"-view_b");CHKERRQ(ierr);
        ierr = VecViewFromOptions(b2,NULL,"-view_b");CHKERRQ(ierr);
        ierr = VecDuplicate(b1,&b);CHKERRQ(ierr);
        ierr = VecCopy(b1,b);CHKERRQ(ierr);
        ierr = VecAXPY(b,-1.0,b2);CHKERRQ(ierr);
        ierr = VecNorm(b,NORM_INFINITY,&err);CHKERRQ(ierr);
        if (err >= tol) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error b %g\n",test,(double)err);CHKERRQ(ierr);
        }
        ierr = VecDestroy(&b);CHKERRQ(ierr);
      }
      ierr = VecDestroy(&b1);CHKERRQ(ierr);
      ierr = VecDestroy(&b2);CHKERRQ(ierr);
      ierr = VecDestroy(&x);CHKERRQ(ierr);
      ierr = MatConvert(S,MATDENSE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);

      ierr = MatCreateTranspose(S,&ST);CHKERRQ(ierr);
      ierr = MatComputeOperator(ST,MATDENSE,&BT);CHKERRQ(ierr);
      ierr = MatTranspose(BT,MAT_INITIAL_MATRIX,&BTT);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)B,"S");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)BTT,"STT");CHKERRQ(ierr);
      ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)C,"A");CHKERRQ(ierr);

      ierr = MatViewFromOptions(C,NULL,"-view_mat");CHKERRQ(ierr);
      ierr = MatViewFromOptions(B,NULL,"-view_mat");CHKERRQ(ierr);
      ierr = MatViewFromOptions(BTT,NULL,"-view_mat");CHKERRQ(ierr);

      ierr = MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(C,NORM_FROBENIUS,&err);CHKERRQ(ierr);
      if (err >= tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat mult after %s %g\n",test,ronl ? "MatZeroRows" : "MatZeroRowsColumns",(double)err);CHKERRQ(ierr);
      }

      ierr = MatConvert(A,MATDENSE,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr);
      ierr = MatAXPY(C,-1.0,BTT,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(C,NORM_FROBENIUS,&err);CHKERRQ(ierr);
      if (err >= tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat mult transpose after %s %g\n",test,ronl ? "MatZeroRows" : "MatZeroRowsColumns",(double)err);CHKERRQ(ierr);
      }

      ierr = MatDestroy(&ST);CHKERRQ(ierr);
      ierr = MatDestroy(&BTT);CHKERRQ(ierr);
      ierr = MatDestroy(&BT);CHKERRQ(ierr);
      ierr = MatDestroy(&B);CHKERRQ(ierr);
      ierr = MatDestroy(&C);CHKERRQ(ierr);
    }
    if (testdiagscale) { /* MatDiagonalScale() */
      Vec vr,vl;

      ierr = MatCreateVecs(A,&vr,&vl);CHKERRQ(ierr);
      if (randomize) {
        ierr = VecSetRandom(vr,NULL);CHKERRQ(ierr);
        ierr = VecSetRandom(vl,NULL);CHKERRQ(ierr);
      } else {
        ierr = VecSet(vr,test%2 ? 0.15 : 1.0/0.15);CHKERRQ(ierr);
        ierr = VecSet(vl,test%2 ? -1.2 : 1.0/-1.2);CHKERRQ(ierr);
      }
      ierr = MatDiagonalScale(A,vl,vr);CHKERRQ(ierr);
      ierr = MatDiagonalScale(S,vl,vr);CHKERRQ(ierr);
      ierr = VecDestroy(&vr);CHKERRQ(ierr);
      ierr = VecDestroy(&vl);CHKERRQ(ierr);
    }

    if (testscale) { /* MatScale() */
      ierr = MatScale(A,test%2 ? 1.4 : 1.0/1.4);CHKERRQ(ierr);
      ierr = MatScale(S,test%2 ? 1.4 : 1.0/1.4);CHKERRQ(ierr);
    }

    if (testshift && cong) { /* MatShift() : MATSHELL shift is broken when row/cols layout are not congruent and left/right scaling have been applied */
      ierr = MatShift(A,test%2 ? -77.5 : 77.5);CHKERRQ(ierr);
      ierr = MatShift(S,test%2 ? -77.5 : 77.5);CHKERRQ(ierr);
    }

    if (testgetdiag && cong) { /* MatGetDiagonal() */
      Vec dA,dS;

      ierr = MatCreateVecs(A,&dA,NULL);CHKERRQ(ierr);
      ierr = MatCreateVecs(S,&dS,NULL);CHKERRQ(ierr);
      ierr = MatGetDiagonal(A,dA);CHKERRQ(ierr);
      ierr = MatGetDiagonal(S,dS);CHKERRQ(ierr);
      ierr = VecAXPY(dA,-1.0,dS);CHKERRQ(ierr);
      ierr = VecNorm(dA,NORM_INFINITY,&err);CHKERRQ(ierr);
      if (err >= tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error diag %g\n",test,(double)err);CHKERRQ(ierr);
      }
      ierr = VecDestroy(&dA);CHKERRQ(ierr);
      ierr = VecDestroy(&dS);CHKERRQ(ierr);
    }

    if (testdup && !test) {
      Mat A2, S2;

      ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
      ierr = MatDuplicate(S,MAT_COPY_VALUES,&S2);CHKERRQ(ierr);
      ierr = MatDestroy(&A);CHKERRQ(ierr);
      ierr = MatDestroy(&S);CHKERRQ(ierr);
      A = A2;
      S = S2;
    }

    if (testsubmat) {
      Mat      sA,sS,dA,dS,At,St;
      IS       r,c;
      PetscInt rst,ren,cst,cen;

      ierr = MatGetOwnershipRange(A,&rst,&ren);CHKERRQ(ierr);
      ierr = MatGetOwnershipRangeColumn(A,&cst,&cen);CHKERRQ(ierr);
      ierr = ISCreateStride(PetscObjectComm((PetscObject)A),(ren-rst)/2,rst,1,&r);CHKERRQ(ierr);
      ierr = ISCreateStride(PetscObjectComm((PetscObject)A),(cen-cst)/2,cst,1,&c);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(A,r,c,MAT_INITIAL_MATRIX,&sA);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(S,r,c,MAT_INITIAL_MATRIX,&sS);CHKERRQ(ierr);
      ierr = MatMultAddEqual(sA,sS,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error submatrix mult add\n",test);CHKERRQ(ierr);
      }
      ierr = MatMultTransposeAddEqual(sA,sS,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error submatrix mult add (T)\n",test);CHKERRQ(ierr);
      }
      ierr = MatConvert(sA,MATDENSE,MAT_INITIAL_MATRIX,&dA);CHKERRQ(ierr);
      ierr = MatConvert(sS,MATDENSE,MAT_INITIAL_MATRIX,&dS);CHKERRQ(ierr);
      ierr = MatAXPY(dA,-1.0,dS,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(dA,NORM_FROBENIUS,&err);CHKERRQ(ierr);
      if (err >= tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat submatrix %g\n",test,(double)err);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&sA);CHKERRQ(ierr);
      ierr = MatDestroy(&sS);CHKERRQ(ierr);
      ierr = MatDestroy(&dA);CHKERRQ(ierr);
      ierr = MatDestroy(&dS);CHKERRQ(ierr);
      ierr = MatCreateTranspose(A,&At);CHKERRQ(ierr);
      ierr = MatCreateTranspose(S,&St);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(At,c,r,MAT_INITIAL_MATRIX,&sA);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(St,c,r,MAT_INITIAL_MATRIX,&sS);CHKERRQ(ierr);
      ierr = MatMultAddEqual(sA,sS,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error submatrix (T) mult add\n",test);CHKERRQ(ierr);
      }
      ierr = MatMultTransposeAddEqual(sA,sS,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error submatrix (T) mult add (T)\n",test);CHKERRQ(ierr);
      }
      ierr = MatConvert(sA,MATDENSE,MAT_INITIAL_MATRIX,&dA);CHKERRQ(ierr);
      ierr = MatConvert(sS,MATDENSE,MAT_INITIAL_MATRIX,&dS);CHKERRQ(ierr);
      ierr = MatAXPY(dA,-1.0,dS,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(dA,NORM_FROBENIUS,&err);CHKERRQ(ierr);
      if (err >= tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat submatrix (T) %g\n",test,(double)err);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&sA);CHKERRQ(ierr);
      ierr = MatDestroy(&sS);CHKERRQ(ierr);
      ierr = MatDestroy(&dA);CHKERRQ(ierr);
      ierr = MatDestroy(&dS);CHKERRQ(ierr);
      ierr = MatDestroy(&At);CHKERRQ(ierr);
      ierr = MatDestroy(&St);CHKERRQ(ierr);
      ierr = ISDestroy(&r);CHKERRQ(ierr);
      ierr = ISDestroy(&c);CHKERRQ(ierr);
    }

    if (testaxpy) {
      Mat          tA,tS,dA,dS;
      MatStructure str[3] = { SAME_NONZERO_PATTERN, SUBSET_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN };

      ierr = MatDuplicate(A,MAT_COPY_VALUES,&tA);CHKERRQ(ierr);
      if (testaxpyd && !(test%2)) {
        ierr = PetscObjectReference((PetscObject)tA);CHKERRQ(ierr);
        tS   = tA;
      } else {
        ierr = PetscObjectReference((PetscObject)S);CHKERRQ(ierr);
        tS   = S;
      }
      ierr = MatAXPY(A,0.5,tA,str[test%3]);CHKERRQ(ierr);
      ierr = MatAXPY(S,0.5,tS,str[test%3]);CHKERRQ(ierr);
      /* this will trigger an error the next MatMult or MatMultTranspose call for S */
      if (testaxpyerr) { ierr = MatScale(tA,0);CHKERRQ(ierr); }
      ierr = MatDestroy(&tA);CHKERRQ(ierr);
      ierr = MatDestroy(&tS);CHKERRQ(ierr);
      ierr = MatMultAddEqual(A,S,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error axpy mult add\n",test);CHKERRQ(ierr);
      }
      ierr = MatMultTransposeAddEqual(A,S,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error axpy mult add (T)\n",test);CHKERRQ(ierr);
      }
      ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&dA);CHKERRQ(ierr);
      ierr = MatConvert(S,MATDENSE,MAT_INITIAL_MATRIX,&dS);CHKERRQ(ierr);
      ierr = MatAXPY(dA,-1.0,dS,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(dA,NORM_FROBENIUS,&err);CHKERRQ(ierr);
      if (err >= tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat submatrix %g\n",test,(double)err);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&dA);CHKERRQ(ierr);
      ierr = MatDestroy(&dS);CHKERRQ(ierr);
    }

    if (testreset && (ntest == 1 || test == ntest-2)) {
      /* reset MATSHELL */
      ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      /* reset A */
      ierr = MatCopy(user->B,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
     suffix: rect
     requires: !single
     output_file: output/ex221_1.out
     nsize: {{1 3}}
     args: -loop 3 -keep {{0 1}} -M {{12 19}} -N {{19 12}} -submat {{0 1}} -test_axpy_different {{0 1}}

   testset:
     suffix: square
     requires: !single
     output_file: output/ex221_1.out
     nsize: {{1 3}}
     args: -M 21 -N 21 -loop 4 -rows_only {{0 1}} -keep {{0 1}} -submat {{0 1}} -test_axpy_different {{0 1}}
TEST*/
