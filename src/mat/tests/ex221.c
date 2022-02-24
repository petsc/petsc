static char help[] = "Tests various routines for MATSHELL\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat B;
};

static PetscErrorCode MatGetDiagonal_User(Mat A,Vec X)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatGetDiagonal(user->B,X));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_User(Mat A,Vec X,Vec Y)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatMult(user->B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_User(Mat A,Vec X,Vec Y)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatMultTranspose(user->B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_User(Mat A,Mat X,MatStructure str)
{
  User           user,userX;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(MatShellGetContext(X,&userX));
  PetscCheckFalse(user != userX,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"This should not happen");
  CHKERRQ(PetscObjectReference((PetscObject)user->B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_User(Mat A)
{
  User           user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&user));
  CHKERRQ(PetscObjectDereference((PetscObject)user->B));
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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ml",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nl",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-square_nc",&nc,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-rows_only",&ronl,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-randomize",&randomize,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-submat",&submat,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_zerorows",&testzerorows,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_diagscale",&testdiagscale,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_getdiag",&testgetdiag,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_shift",&testshift,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_scale",&testscale,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_dup",&testdup,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_reset",&testreset,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_submat",&testsubmat,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_axpy",&testaxpy,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_axpy_different",&testaxpyd,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_axpy_error",&testaxpyerr,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-loop",&ntest,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tol",&tol,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-diag",&diag,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-keep",&keep,NULL));
  /* This tests square matrices with different row/col layout */
  if (nc && size > 1) {
    M = PetscMax(PetscMax(N,M),1);
    N = M;
    m = n = 0;
    if (rank == 0) { m = M-1; n = 1; }
    else if (rank == 1) { m = 1; n = N-1; }
  }
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,m,n,M,N,NULL,&A));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetOwnershipRange(A,&s1,NULL));
  s2   = 1;
  while (s2 < M) s2 *= 10;
  CHKERRQ(MatDenseGetArray(A,&data));
  for (j = 0; j < N; j++) {
    for (i = 0; i < m; i++) {
      data[j*m + i] = s2*j + i + s1 + 1;
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  if (submat) {
    Mat      A2;
    IS       r,c;
    PetscInt rst,ren,cst,cen;

    CHKERRQ(MatGetOwnershipRange(A,&rst,&ren));
    CHKERRQ(MatGetOwnershipRangeColumn(A,&cst,&cen));
    CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)A),(ren-rst)/2,rst,1,&r));
    CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)A),(cen-cst)/2,cst,1,&c));
    CHKERRQ(MatCreateSubMatrix(A,r,c,MAT_INITIAL_MATRIX,&A2));
    CHKERRQ(ISDestroy(&r));
    CHKERRQ(ISDestroy(&c));
    CHKERRQ(MatDestroy(&A));
    A = A2;
  }

  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(MatHasCongruentLayouts(A,&cong));

  CHKERRQ(MatConvert(A,MATAIJ,MAT_INPLACE_MATRIX,&A));
  CHKERRQ(MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,keep));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"initial"));
  CHKERRQ(MatViewFromOptions(A,NULL,"-view_mat"));

  CHKERRQ(PetscNew(&user));
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,m,n,M,N,user,&S));
  CHKERRQ(MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MatMult_User));
  CHKERRQ(MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_User));
  if (cong) {
    CHKERRQ(MatShellSetOperation(S,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_User));
  }
  CHKERRQ(MatShellSetOperation(S,MATOP_COPY,(void (*)(void))MatCopy_User));
  CHKERRQ(MatShellSetOperation(S,MATOP_DESTROY,(void (*)(void))MatDestroy_User));
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&user->B));

  /* Square and rows only scaling */
  ronl = cong ? ronl : PETSC_TRUE;

  for (test = 0; test < ntest; test++) {
    PetscReal err;

    CHKERRQ(MatMultAddEqual(A,S,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mult add\n",test));
    }
    CHKERRQ(MatMultTransposeAddEqual(A,S,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mult add (T)\n",test));
    }
    if (testzerorows) {
      Mat       ST,B,C,BT,BTT;
      IS        zr;
      Vec       x = NULL, b1 = NULL, b2 = NULL;
      PetscInt  *idxs = NULL, nr = 0;

      if (rank == (test%size)) {
        nr = 1;
        CHKERRQ(PetscMalloc1(nr,&idxs));
        if (test%2) {
          idxs[0] = (2*M - 1 - test/2)%M;
        } else {
          idxs[0] = (test/2)%M;
        }
        idxs[0] = PetscMax(idxs[0],0);
      }
      CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,nr,idxs,PETSC_OWN_POINTER,&zr));
      CHKERRQ(PetscObjectSetName((PetscObject)zr,"ZR"));
      CHKERRQ(ISViewFromOptions(zr,NULL,"-view_is"));
      CHKERRQ(MatCreateVecs(A,&x,&b1));
      if (randomize) {
        CHKERRQ(VecSetRandom(x,NULL));
        CHKERRQ(VecSetRandom(b1,NULL));
      } else {
        CHKERRQ(VecSet(x,11.4));
        CHKERRQ(VecSet(b1,-14.2));
      }
      CHKERRQ(VecDuplicate(b1,&b2));
      CHKERRQ(VecCopy(b1,b2));
      CHKERRQ(PetscObjectSetName((PetscObject)b1,"A_B1"));
      CHKERRQ(PetscObjectSetName((PetscObject)b2,"A_B2"));
      if (size > 1 && !cong) { /* MATMPIAIJ ZeroRows and ZeroRowsColumns are buggy in this case */
        CHKERRQ(VecDestroy(&b1));
      }
      if (ronl) {
        CHKERRQ(MatZeroRowsIS(A,zr,diag,x,b1));
        CHKERRQ(MatZeroRowsIS(S,zr,diag,x,b2));
      } else {
        CHKERRQ(MatZeroRowsColumnsIS(A,zr,diag,x,b1));
        CHKERRQ(MatZeroRowsColumnsIS(S,zr,diag,x,b2));
        CHKERRQ(ISDestroy(&zr));
        /* Mix zerorows and zerorowscols */
        nr   = 0;
        idxs = NULL;
        if (rank == 0) {
          nr   = 1;
          CHKERRQ(PetscMalloc1(nr,&idxs));
          if (test%2) {
            idxs[0] = (3*M - 2 - test/2)%M;
          } else {
            idxs[0] = (test/2+1)%M;
          }
          idxs[0] = PetscMax(idxs[0],0);
        }
        CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,nr,idxs,PETSC_OWN_POINTER,&zr));
        CHKERRQ(PetscObjectSetName((PetscObject)zr,"ZR2"));
        CHKERRQ(ISViewFromOptions(zr,NULL,"-view_is"));
        CHKERRQ(MatZeroRowsIS(A,zr,diag*2.0+PETSC_SMALL,NULL,NULL));
        CHKERRQ(MatZeroRowsIS(S,zr,diag*2.0+PETSC_SMALL,NULL,NULL));
      }
      CHKERRQ(ISDestroy(&zr));

      if (b1) {
        Vec b;

        CHKERRQ(VecViewFromOptions(b1,NULL,"-view_b"));
        CHKERRQ(VecViewFromOptions(b2,NULL,"-view_b"));
        CHKERRQ(VecDuplicate(b1,&b));
        CHKERRQ(VecCopy(b1,b));
        CHKERRQ(VecAXPY(b,-1.0,b2));
        CHKERRQ(VecNorm(b,NORM_INFINITY,&err));
        if (err >= tol) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error b %g\n",test,(double)err));
        }
        CHKERRQ(VecDestroy(&b));
      }
      CHKERRQ(VecDestroy(&b1));
      CHKERRQ(VecDestroy(&b2));
      CHKERRQ(VecDestroy(&x));
      CHKERRQ(MatConvert(S,MATDENSE,MAT_INITIAL_MATRIX,&B));

      CHKERRQ(MatCreateTranspose(S,&ST));
      CHKERRQ(MatComputeOperator(ST,MATDENSE,&BT));
      CHKERRQ(MatTranspose(BT,MAT_INITIAL_MATRIX,&BTT));
      CHKERRQ(PetscObjectSetName((PetscObject)B,"S"));
      CHKERRQ(PetscObjectSetName((PetscObject)BTT,"STT"));
      CHKERRQ(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&C));
      CHKERRQ(PetscObjectSetName((PetscObject)C,"A"));

      CHKERRQ(MatViewFromOptions(C,NULL,"-view_mat"));
      CHKERRQ(MatViewFromOptions(B,NULL,"-view_mat"));
      CHKERRQ(MatViewFromOptions(BTT,NULL,"-view_mat"));

      CHKERRQ(MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN));
      CHKERRQ(MatNorm(C,NORM_FROBENIUS,&err));
      if (err >= tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat mult after %s %g\n",test,ronl ? "MatZeroRows" : "MatZeroRowsColumns",(double)err));
      }

      CHKERRQ(MatConvert(A,MATDENSE,MAT_REUSE_MATRIX,&C));
      CHKERRQ(MatAXPY(C,-1.0,BTT,SAME_NONZERO_PATTERN));
      CHKERRQ(MatNorm(C,NORM_FROBENIUS,&err));
      if (err >= tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat mult transpose after %s %g\n",test,ronl ? "MatZeroRows" : "MatZeroRowsColumns",(double)err));
      }

      CHKERRQ(MatDestroy(&ST));
      CHKERRQ(MatDestroy(&BTT));
      CHKERRQ(MatDestroy(&BT));
      CHKERRQ(MatDestroy(&B));
      CHKERRQ(MatDestroy(&C));
    }
    if (testdiagscale) { /* MatDiagonalScale() */
      Vec vr,vl;

      CHKERRQ(MatCreateVecs(A,&vr,&vl));
      if (randomize) {
        CHKERRQ(VecSetRandom(vr,NULL));
        CHKERRQ(VecSetRandom(vl,NULL));
      } else {
        CHKERRQ(VecSet(vr,test%2 ? 0.15 : 1.0/0.15));
        CHKERRQ(VecSet(vl,test%2 ? -1.2 : 1.0/-1.2));
      }
      CHKERRQ(MatDiagonalScale(A,vl,vr));
      CHKERRQ(MatDiagonalScale(S,vl,vr));
      CHKERRQ(VecDestroy(&vr));
      CHKERRQ(VecDestroy(&vl));
    }

    if (testscale) { /* MatScale() */
      CHKERRQ(MatScale(A,test%2 ? 1.4 : 1.0/1.4));
      CHKERRQ(MatScale(S,test%2 ? 1.4 : 1.0/1.4));
    }

    if (testshift && cong) { /* MatShift() : MATSHELL shift is broken when row/cols layout are not congruent and left/right scaling have been applied */
      CHKERRQ(MatShift(A,test%2 ? -77.5 : 77.5));
      CHKERRQ(MatShift(S,test%2 ? -77.5 : 77.5));
    }

    if (testgetdiag && cong) { /* MatGetDiagonal() */
      Vec dA,dS;

      CHKERRQ(MatCreateVecs(A,&dA,NULL));
      CHKERRQ(MatCreateVecs(S,&dS,NULL));
      CHKERRQ(MatGetDiagonal(A,dA));
      CHKERRQ(MatGetDiagonal(S,dS));
      CHKERRQ(VecAXPY(dA,-1.0,dS));
      CHKERRQ(VecNorm(dA,NORM_INFINITY,&err));
      if (err >= tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error diag %g\n",test,(double)err));
      }
      CHKERRQ(VecDestroy(&dA));
      CHKERRQ(VecDestroy(&dS));
    }

    if (testdup && !test) {
      Mat A2, S2;

      CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));
      CHKERRQ(MatDuplicate(S,MAT_COPY_VALUES,&S2));
      CHKERRQ(MatDestroy(&A));
      CHKERRQ(MatDestroy(&S));
      A = A2;
      S = S2;
    }

    if (testsubmat) {
      Mat      sA,sS,dA,dS,At,St;
      IS       r,c;
      PetscInt rst,ren,cst,cen;

      CHKERRQ(MatGetOwnershipRange(A,&rst,&ren));
      CHKERRQ(MatGetOwnershipRangeColumn(A,&cst,&cen));
      CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)A),(ren-rst)/2,rst,1,&r));
      CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)A),(cen-cst)/2,cst,1,&c));
      CHKERRQ(MatCreateSubMatrix(A,r,c,MAT_INITIAL_MATRIX,&sA));
      CHKERRQ(MatCreateSubMatrix(S,r,c,MAT_INITIAL_MATRIX,&sS));
      CHKERRQ(MatMultAddEqual(sA,sS,10,&flg));
      if (!flg) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error submatrix mult add\n",test));
      }
      CHKERRQ(MatMultTransposeAddEqual(sA,sS,10,&flg));
      if (!flg) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error submatrix mult add (T)\n",test));
      }
      CHKERRQ(MatConvert(sA,MATDENSE,MAT_INITIAL_MATRIX,&dA));
      CHKERRQ(MatConvert(sS,MATDENSE,MAT_INITIAL_MATRIX,&dS));
      CHKERRQ(MatAXPY(dA,-1.0,dS,SAME_NONZERO_PATTERN));
      CHKERRQ(MatNorm(dA,NORM_FROBENIUS,&err));
      if (err >= tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat submatrix %g\n",test,(double)err));
      }
      CHKERRQ(MatDestroy(&sA));
      CHKERRQ(MatDestroy(&sS));
      CHKERRQ(MatDestroy(&dA));
      CHKERRQ(MatDestroy(&dS));
      CHKERRQ(MatCreateTranspose(A,&At));
      CHKERRQ(MatCreateTranspose(S,&St));
      CHKERRQ(MatCreateSubMatrix(At,c,r,MAT_INITIAL_MATRIX,&sA));
      CHKERRQ(MatCreateSubMatrix(St,c,r,MAT_INITIAL_MATRIX,&sS));
      CHKERRQ(MatMultAddEqual(sA,sS,10,&flg));
      if (!flg) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error submatrix (T) mult add\n",test));
      }
      CHKERRQ(MatMultTransposeAddEqual(sA,sS,10,&flg));
      if (!flg) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error submatrix (T) mult add (T)\n",test));
      }
      CHKERRQ(MatConvert(sA,MATDENSE,MAT_INITIAL_MATRIX,&dA));
      CHKERRQ(MatConvert(sS,MATDENSE,MAT_INITIAL_MATRIX,&dS));
      CHKERRQ(MatAXPY(dA,-1.0,dS,SAME_NONZERO_PATTERN));
      CHKERRQ(MatNorm(dA,NORM_FROBENIUS,&err));
      if (err >= tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat submatrix (T) %g\n",test,(double)err));
      }
      CHKERRQ(MatDestroy(&sA));
      CHKERRQ(MatDestroy(&sS));
      CHKERRQ(MatDestroy(&dA));
      CHKERRQ(MatDestroy(&dS));
      CHKERRQ(MatDestroy(&At));
      CHKERRQ(MatDestroy(&St));
      CHKERRQ(ISDestroy(&r));
      CHKERRQ(ISDestroy(&c));
    }

    if (testaxpy) {
      Mat          tA,tS,dA,dS;
      MatStructure str[3] = { SAME_NONZERO_PATTERN, SUBSET_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN };

      CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&tA));
      if (testaxpyd && !(test%2)) {
        CHKERRQ(PetscObjectReference((PetscObject)tA));
        tS   = tA;
      } else {
        CHKERRQ(PetscObjectReference((PetscObject)S));
        tS   = S;
      }
      CHKERRQ(MatAXPY(A,0.5,tA,str[test%3]));
      CHKERRQ(MatAXPY(S,0.5,tS,str[test%3]));
      /* this will trigger an error the next MatMult or MatMultTranspose call for S */
      if (testaxpyerr) CHKERRQ(MatScale(tA,0));
      CHKERRQ(MatDestroy(&tA));
      CHKERRQ(MatDestroy(&tS));
      CHKERRQ(MatMultAddEqual(A,S,10,&flg));
      if (!flg) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error axpy mult add\n",test));
      }
      CHKERRQ(MatMultTransposeAddEqual(A,S,10,&flg));
      if (!flg) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error axpy mult add (T)\n",test));
      }
      CHKERRQ(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&dA));
      CHKERRQ(MatConvert(S,MATDENSE,MAT_INITIAL_MATRIX,&dS));
      CHKERRQ(MatAXPY(dA,-1.0,dS,SAME_NONZERO_PATTERN));
      CHKERRQ(MatNorm(dA,NORM_FROBENIUS,&err));
      if (err >= tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[test %" PetscInt_FMT "] Error mat submatrix %g\n",test,(double)err));
      }
      CHKERRQ(MatDestroy(&dA));
      CHKERRQ(MatDestroy(&dS));
    }

    if (testreset && (ntest == 1 || test == ntest-2)) {
      /* reset MATSHELL */
      CHKERRQ(MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY));
      /* reset A */
      CHKERRQ(MatCopy(user->B,A,DIFFERENT_NONZERO_PATTERN));
    }
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&S));
  CHKERRQ(PetscFree(user));
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
