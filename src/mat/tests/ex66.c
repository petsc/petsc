static char help[] = "Tests MATH2OPUS\n\n";

#include <petscmat.h>
#include <petscsf.h>

static PetscScalar GenEntry_Symm(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
    PetscInt  d;
    PetscReal clength = sdim == 3 ? 0.2 : 0.1;
    PetscReal dist, diff = 0.0;

    for (d = 0; d < sdim; d++) { diff += (x[d] - y[d]) * (x[d] - y[d]); }
    dist = PetscSqrtReal(diff);
    return PetscExpReal(-dist / clength);
}

static PetscScalar GenEntry_Unsymm(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
    PetscInt  d;
    PetscReal clength = sdim == 3 ? 0.2 : 0.1;
    PetscReal dist, diff = 0.0, nx = 0.0, ny = 0.0;

    for (d = 0; d < sdim; d++) { nx += x[d]*x[d]; }
    for (d = 0; d < sdim; d++) { ny += y[d]*y[d]; }
    for (d = 0; d < sdim; d++) { diff += (x[d] - y[d]) * (x[d] - y[d]); }
    dist = PetscSqrtReal(diff);
    return nx > ny ? PetscExpReal(-dist / clength) : PetscExpReal(-dist / clength) + 1.;
}

int main(int argc,char **argv)
{
  Mat            A,B,C,D;
  Vec            v,x,y,Ax,Ay,Bx,By;
  PetscRandom    r;
  PetscLayout    map;
  PetscScalar    *Adata = NULL, *Cdata = NULL, scale = 1.0;
  PetscReal      *coords,nA,nD,nB,err,nX,norms[3];
  PetscInt       N, n = 64, dim = 1, i, j, nrhs = 11, lda = 0, ldc = 0, ldu = 0, nlr = 7, nt, ntrials = 2;
  PetscMPIInt    size,rank;
  PetscBool      testlayout = PETSC_FALSE, flg, symm = PETSC_FALSE, Asymm = PETSC_TRUE, kernel = PETSC_TRUE;
  PetscBool      checkexpl = PETSC_FALSE, agpu = PETSC_FALSE, bgpu = PETSC_FALSE, cgpu = PETSC_FALSE, flgglob;
  PetscBool      testtrans, testnorm, randommat = PETSC_TRUE, testorthog, testcompress, testhlru;
  void           (*approxnormfunc)(void);
  void           (*Anormfunc)(void);
  PetscErrorCode ierr;

#if defined(PETSC_HAVE_MPI_INIT_THREAD)
  PETSC_MPI_THREAD_REQUIRED = MPI_THREAD_MULTIPLE;
#endif
  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ng",&N,&flgglob));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-lda",&lda,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ldc",&ldc,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nlr",&nlr,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ldu",&ldu,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-matmattrials",&ntrials,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-randommat",&randommat,NULL));
  if (!flgglob) CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testlayout",&testlayout,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-Asymm",&Asymm,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-symm",&symm,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-kernel",&kernel,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-checkexpl",&checkexpl,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-agpu",&agpu,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-bgpu",&bgpu,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-cgpu",&cgpu,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-scale",&scale,NULL));
  if (!Asymm) symm = PETSC_FALSE;

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Disable tests for unimplemented variants */
  testtrans = (PetscBool)(size == 1 || symm);
  testnorm = (PetscBool)(size == 1 || symm);
  testorthog = (PetscBool)(size == 1 || symm);
  testcompress = (PetscBool)(size == 1 || symm);
  testhlru = (PetscBool)(size == 1);

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscLayoutCreate(PETSC_COMM_WORLD,&map));
  if (testlayout) {
    if (rank%2) n = PetscMax(2*n-5*rank,0);
    else n = 2*n+rank;
  }
  if (!flgglob) {
    CHKERRQ(PetscLayoutSetLocalSize(map,n));
    CHKERRQ(PetscLayoutSetUp(map));
    CHKERRQ(PetscLayoutGetSize(map,&N));
  } else {
    CHKERRQ(PetscLayoutSetSize(map,N));
    CHKERRQ(PetscLayoutSetUp(map));
    CHKERRQ(PetscLayoutGetLocalSize(map,&n));
  }
  CHKERRQ(PetscLayoutDestroy(&map));

  if (lda) {
    CHKERRQ(PetscMalloc1(N*(n+lda),&Adata));
  }
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,n,n,N,N,Adata,&A));
  CHKERRQ(MatDenseSetLDA(A,n+lda));

  /* Create random points; these are replicated in order to populate a dense matrix and to compare sequential and dense runs
     The constructor for MATH2OPUS can take as input the distributed coordinates and replicates them internally in case
     a kernel construction is requested */
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  CHKERRQ(PetscRandomSetFromOptions(r));
  CHKERRQ(PetscRandomSetSeed(r,123456));
  CHKERRQ(PetscRandomSeed(r));
  CHKERRQ(PetscMalloc1(N*dim,&coords));
  CHKERRQ(PetscRandomGetValuesReal(r,N*dim,coords));
  CHKERRQ(PetscRandomDestroy(&r));

  if (kernel || !randommat) {
    MatH2OpusKernel k = Asymm ? GenEntry_Symm : GenEntry_Unsymm;
    PetscInt        ist,ien;

    CHKERRQ(MatGetOwnershipRange(A,&ist,&ien));
    for (i = ist; i < ien; i++) {
      for (j = 0; j < N; j++) {
        CHKERRQ(MatSetValue(A,i,j,(*k)(dim,coords + i*dim,coords + j*dim,NULL),INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    if (kernel) {
      CHKERRQ(MatCreateH2OpusFromKernel(PETSC_COMM_WORLD,n,n,N,N,dim,coords + ist*dim,PETSC_TRUE,k,NULL,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&B));
    } else {
      CHKERRQ(MatCreateH2OpusFromMat(A,dim,coords + ist*dim,PETSC_TRUE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&B));
    }
  } else {
    PetscInt ist;

    CHKERRQ(MatGetOwnershipRange(A,&ist,NULL));
    CHKERRQ(MatSetRandom(A,NULL));
    if (Asymm) {
      CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&B));
      CHKERRQ(MatAXPY(A,1.0,B,SAME_NONZERO_PATTERN));
      CHKERRQ(MatDestroy(&B));
      CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
    }
    CHKERRQ(MatCreateH2OpusFromMat(A,dim,coords + ist*dim,PETSC_TRUE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&B));
  }
  CHKERRQ(PetscFree(coords));
  if (agpu) {
    CHKERRQ(MatConvert(A,MATDENSECUDA,MAT_INPLACE_MATRIX,&A));
  }
  CHKERRQ(MatViewFromOptions(A,NULL,"-A_view"));

  CHKERRQ(MatSetOption(B,MAT_SYMMETRIC,symm));

  /* assemble the H-matrix */
  CHKERRQ(MatBindToCPU(B,(PetscBool)!bgpu));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatViewFromOptions(B,NULL,"-B_view"));

  /* Test MatScale */
  CHKERRQ(MatScale(A,scale));
  CHKERRQ(MatScale(B,scale));

  /* Test MatMult */
  CHKERRQ(MatCreateVecs(A,&Ax,&Ay));
  CHKERRQ(MatCreateVecs(B,&Bx,&By));
  CHKERRQ(VecSetRandom(Ax,NULL));
  CHKERRQ(VecCopy(Ax,Bx));
  CHKERRQ(MatMult(A,Ax,Ay));
  CHKERRQ(MatMult(B,Bx,By));
  CHKERRQ(VecViewFromOptions(Ay,NULL,"-mult_vec_view"));
  CHKERRQ(VecViewFromOptions(By,NULL,"-mult_vec_view"));
  CHKERRQ(VecNorm(Ay,NORM_INFINITY,&nX));
  CHKERRQ(VecAXPY(Ay,-1.0,By));
  CHKERRQ(VecViewFromOptions(Ay,NULL,"-mult_vec_view"));
  CHKERRQ(VecNorm(Ay,NORM_INFINITY,&err));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMult err %g\n",err/nX));
  CHKERRQ(VecScale(By,-1.0));
  CHKERRQ(MatMultAdd(B,Bx,By,By));
  CHKERRQ(VecNorm(By,NORM_INFINITY,&err));
  CHKERRQ(VecViewFromOptions(By,NULL,"-mult_vec_view"));
  if (err > 10.*PETSC_SMALL) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMultAdd err %g\n",err));
  }

  /* Test MatNorm */
  CHKERRQ(MatNorm(A,NORM_INFINITY,&norms[0]));
  CHKERRQ(MatNorm(A,NORM_1,&norms[1]));
  norms[2] = -1.; /* NORM_2 not supported */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"A Matrix norms:        infty=%g, norm_1=%g, norm_2=%g\n",(double)norms[0],(double)norms[1],(double)norms[2]));
  CHKERRQ(MatGetOperation(A,MATOP_NORM,&Anormfunc));
  CHKERRQ(MatGetOperation(B,MATOP_NORM,&approxnormfunc));
  CHKERRQ(MatSetOperation(A,MATOP_NORM,approxnormfunc));
  CHKERRQ(MatNorm(A,NORM_INFINITY,&norms[0]));
  CHKERRQ(MatNorm(A,NORM_1,&norms[1]));
  CHKERRQ(MatNorm(A,NORM_2,&norms[2]));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"A Approx Matrix norms: infty=%g, norm_1=%g, norm_2=%g\n",(double)norms[0],(double)norms[1],(double)norms[2]));
  if (testnorm) {
    CHKERRQ(MatNorm(B,NORM_INFINITY,&norms[0]));
    CHKERRQ(MatNorm(B,NORM_1,&norms[1]));
    CHKERRQ(MatNorm(B,NORM_2,&norms[2]));
  } else {
    norms[0] = -1.;
    norms[1] = -1.;
    norms[2] = -1.;
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"B Approx Matrix norms: infty=%g, norm_1=%g, norm_2=%g\n",(double)norms[0],(double)norms[1],(double)norms[2]));
  CHKERRQ(MatSetOperation(A,MATOP_NORM,Anormfunc));

  /* Test MatDuplicate */
  CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&D));
  CHKERRQ(MatSetOption(D,MAT_SYMMETRIC,symm));
  CHKERRQ(MatMultEqual(B,D,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMult error after MatDuplicate\n"));
  }
  if (testtrans) {
    CHKERRQ(MatMultTransposeEqual(B,D,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMultTranspose error after MatDuplicate\n"));
    }
  }
  CHKERRQ(MatDestroy(&D));

  if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
    CHKERRQ(VecSetRandom(Ay,NULL));
    CHKERRQ(VecCopy(Ay,By));
    CHKERRQ(MatMultTranspose(A,Ay,Ax));
    CHKERRQ(MatMultTranspose(B,By,Bx));
    CHKERRQ(VecViewFromOptions(Ax,NULL,"-multtrans_vec_view"));
    CHKERRQ(VecViewFromOptions(Bx,NULL,"-multtrans_vec_view"));
    CHKERRQ(VecNorm(Ax,NORM_INFINITY,&nX));
    CHKERRQ(VecAXPY(Ax,-1.0,Bx));
    CHKERRQ(VecViewFromOptions(Ax,NULL,"-multtrans_vec_view"));
    CHKERRQ(VecNorm(Ax,NORM_INFINITY,&err));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMultTranspose err %g\n",err/nX));
    CHKERRQ(VecScale(Bx,-1.0));
    CHKERRQ(MatMultTransposeAdd(B,By,Bx,Bx));
    CHKERRQ(VecNorm(Bx,NORM_INFINITY,&err));
    if (err > 10.*PETSC_SMALL) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMultTransposeAdd err %g\n",err));
    }
  }
  CHKERRQ(VecDestroy(&Ax));
  CHKERRQ(VecDestroy(&Ay));
  CHKERRQ(VecDestroy(&Bx));
  CHKERRQ(VecDestroy(&By));

  /* Test MatMatMult */
  if (ldc) {
    CHKERRQ(PetscMalloc1(nrhs*(n+ldc),&Cdata));
  }
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,N,nrhs,Cdata,&C));
  CHKERRQ(MatDenseSetLDA(C,n+ldc));
  CHKERRQ(MatSetRandom(C,NULL));
  if (cgpu) {
    CHKERRQ(MatConvert(C,MATDENSECUDA,MAT_INPLACE_MATRIX,&C));
  }
  for (nt = 0; nt < ntrials; nt++) {
    CHKERRQ(MatMatMult(B,C,nt ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D));
    CHKERRQ(MatViewFromOptions(D,NULL,"-bc_view"));
    CHKERRQ(PetscObjectBaseTypeCompareAny((PetscObject)D,&flg,MATSEQDENSE,MATMPIDENSE,""));
    if (flg) {
      CHKERRQ(MatCreateVecs(B,&x,&y));
      CHKERRQ(MatCreateVecs(D,NULL,&v));
      for (i = 0; i < nrhs; i++) {
        CHKERRQ(MatGetColumnVector(D,v,i));
        CHKERRQ(MatGetColumnVector(C,x,i));
        CHKERRQ(MatMult(B,x,y));
        CHKERRQ(VecAXPY(y,-1.0,v));
        CHKERRQ(VecNorm(y,NORM_INFINITY,&err));
        if (err > 10.*PETSC_SMALL) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMat err %" PetscInt_FMT " %g\n",i,err));
      }
      CHKERRQ(VecDestroy(&y));
      CHKERRQ(VecDestroy(&x));
      CHKERRQ(VecDestroy(&v));
    }
  }
  CHKERRQ(MatDestroy(&D));

  /* Test MatTransposeMatMult */
  if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
    for (nt = 0; nt < ntrials; nt++) {
      CHKERRQ(MatTransposeMatMult(B,C,nt ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D));
      CHKERRQ(MatViewFromOptions(D,NULL,"-btc_view"));
      CHKERRQ(PetscObjectBaseTypeCompareAny((PetscObject)D,&flg,MATSEQDENSE,MATMPIDENSE,""));
      if (flg) {
        CHKERRQ(MatCreateVecs(B,&y,&x));
        CHKERRQ(MatCreateVecs(D,NULL,&v));
        for (i = 0; i < nrhs; i++) {
          CHKERRQ(MatGetColumnVector(D,v,i));
          CHKERRQ(MatGetColumnVector(C,x,i));
          CHKERRQ(MatMultTranspose(B,x,y));
          CHKERRQ(VecAXPY(y,-1.0,v));
          CHKERRQ(VecNorm(y,NORM_INFINITY,&err));
          if (err > 10.*PETSC_SMALL) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatTransMat err %" PetscInt_FMT " %g\n",i,err));
        }
        CHKERRQ(VecDestroy(&y));
        CHKERRQ(VecDestroy(&x));
        CHKERRQ(VecDestroy(&v));
      }
    }
    CHKERRQ(MatDestroy(&D));
  }

  /* Test basis orthogonalization */
  if (testorthog) {
    CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&D));
    CHKERRQ(MatSetOption(D,MAT_SYMMETRIC,symm));
    CHKERRQ(MatH2OpusOrthogonalize(D));
    CHKERRQ(MatMultEqual(B,D,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMult error after basis ortogonalization\n"));
    }
    CHKERRQ(MatDestroy(&D));
  }

  /* Test matrix compression */
  if (testcompress) {
    CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&D));
    CHKERRQ(MatSetOption(D,MAT_SYMMETRIC,symm));
    CHKERRQ(MatH2OpusCompress(D,PETSC_SMALL));
    CHKERRQ(MatDestroy(&D));
  }

  /* Test low-rank update */
  if (testhlru) {
    Mat         U, V;
    PetscScalar *Udata = NULL, *Vdata = NULL;

    if (ldu) {
      CHKERRQ(PetscMalloc1(nlr*(n+ldu),&Udata));
      CHKERRQ(PetscMalloc1(nlr*(n+ldu+2),&Vdata));
    }
    CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&D));
    CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)D),n,PETSC_DECIDE,N,nlr,Udata,&U));
    CHKERRQ(MatDenseSetLDA(U,n+ldu));
    CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)D),n,PETSC_DECIDE,N,nlr,Vdata,&V));
    if (ldu) CHKERRQ(MatDenseSetLDA(V,n+ldu+2));
    CHKERRQ(MatSetRandom(U,NULL));
    CHKERRQ(MatSetRandom(V,NULL));
    CHKERRQ(MatH2OpusLowRankUpdate(D,U,V,0.5));
    CHKERRQ(MatH2OpusLowRankUpdate(D,U,V,-0.5));
    CHKERRQ(MatMultEqual(B,D,10,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMult error after low-rank update\n"));
    }
    CHKERRQ(MatDestroy(&D));
    CHKERRQ(MatDestroy(&U));
    CHKERRQ(PetscFree(Udata));
    CHKERRQ(MatDestroy(&V));
    CHKERRQ(PetscFree(Vdata));
  }

  /* check explicit operator */
  if (checkexpl) {
    Mat Be, Bet;

    CHKERRQ(MatComputeOperator(B,MATDENSE,&D));
    CHKERRQ(MatDuplicate(D,MAT_COPY_VALUES,&Be));
    CHKERRQ(MatNorm(D,NORM_FROBENIUS,&nB));
    CHKERRQ(MatViewFromOptions(D,NULL,"-expl_view"));
    CHKERRQ(MatAXPY(D,-1.0,A,SAME_NONZERO_PATTERN));
    CHKERRQ(MatViewFromOptions(D,NULL,"-diff_view"));
    CHKERRQ(MatNorm(D,NORM_FROBENIUS,&nD));
    CHKERRQ(MatNorm(A,NORM_FROBENIUS,&nA));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Approximation error %g (%g / %g, %g)\n",nD/nA,nD,nA,nB));
    CHKERRQ(MatDestroy(&D));

    if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
      CHKERRQ(MatTranspose(A,MAT_INPLACE_MATRIX,&A));
      CHKERRQ(MatComputeOperatorTranspose(B,MATDENSE,&D));
      CHKERRQ(MatDuplicate(D,MAT_COPY_VALUES,&Bet));
      CHKERRQ(MatNorm(D,NORM_FROBENIUS,&nB));
      CHKERRQ(MatViewFromOptions(D,NULL,"-expl_trans_view"));
      CHKERRQ(MatAXPY(D,-1.0,A,SAME_NONZERO_PATTERN));
      CHKERRQ(MatViewFromOptions(D,NULL,"-diff_trans_view"));
      CHKERRQ(MatNorm(D,NORM_FROBENIUS,&nD));
      CHKERRQ(MatNorm(A,NORM_FROBENIUS,&nA));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Approximation error transpose %g (%g / %g, %g)\n",nD/nA,nD,nA,nB));
      CHKERRQ(MatDestroy(&D));

      CHKERRQ(MatTranspose(Bet,MAT_INPLACE_MATRIX,&Bet));
      CHKERRQ(MatAXPY(Be,-1.0,Bet,SAME_NONZERO_PATTERN));
      CHKERRQ(MatViewFromOptions(Be,NULL,"-diff_expl_view"));
      CHKERRQ(MatNorm(Be,NORM_FROBENIUS,&nB));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Approximation error B - (B^T)^T %g\n",nB));
      CHKERRQ(MatDestroy(&Be));
      CHKERRQ(MatDestroy(&Bet));
    }
  }
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFree(Cdata));
  CHKERRQ(PetscFree(Adata));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: h2opus

#tests from kernel
   test:
     requires: h2opus
     nsize: 1
     suffix: 1
     args: -n {{17 33}} -kernel 1 -dim {{1 2 3}} -symm {{0 1}} -checkexpl -bgpu 0

   test:
     requires: h2opus
     nsize: 1
     suffix: 1_ld
     output_file: output/ex66_1.out
     args: -n 33 -kernel 1 -dim 1 -lda 13 -ldc 11 -ldu 17 -symm 0 -checkexpl -bgpu 0

   test:
     requires: h2opus cuda
     nsize: 1
     suffix: 1_cuda
     output_file: output/ex66_1.out
     args: -n {{17 33}} -kernel 1 -dim {{1 2 3}} -symm {{0 1}} -checkexpl -bgpu 1

   test:
     requires: h2opus cuda
     nsize: 1
     suffix: 1_cuda_ld
     output_file: output/ex66_1.out
     args: -n 33 -kernel 1 -dim 1 -lda 13 -ldc 11 -ldu 17 -symm 0 -checkexpl -bgpu 1

   test:
     requires: h2opus
     nsize: {{2 3}}
     suffix: 1_par
     args: -n 64 -symm -kernel 1 -dim 1 -ldc 12 -testlayout {{0 1}} -bgpu 0 -cgpu 0

   test:
     requires: h2opus cuda
     nsize: {{2 3}}
     suffix: 1_par_cuda
     args: -n 64 -symm -kernel 1 -dim 1 -ldc 12 -testlayout {{0 1}} -bgpu {{0 1}} -cgpu {{0 1}}
     output_file: output/ex66_1_par.out

#tests from matrix sampling (parallel or unsymmetric not supported)
   test:
     requires: h2opus
     nsize: 1
     suffix: 2
     args: -n {{17 33}} -kernel 0 -dim 2 -symm 1 -checkexpl -bgpu 0

   test:
     requires: h2opus cuda
     nsize: 1
     suffix: 2_cuda
     output_file: output/ex66_2.out
     args: -n {{17 33}} -kernel 0 -dim 2 -symm 1 -checkexpl -bgpu {{0 1}} -agpu {{0 1}}

#tests view operation
   test:
     requires: h2opus !cuda
     filter: grep -v "MPI processes" | grep -v "\[" | grep -v "\]"
     nsize: {{1 2 3}}
     suffix: view
     args: -ng 64 -kernel 1 -dim 2 -symm 1 -checkexpl -B_view -mat_h2opus_leafsize 17 -mat_h2opus_normsamples 13 -mat_h2opus_indexmap_view ::ascii_matlab -mat_approximate_norm_samples 2 -mat_h2opus_normsamples 2

   test:
     requires: h2opus cuda
     filter: grep -v "MPI processes" | grep -v "\[" | grep -v "\]"
     nsize: {{1 2 3}}
     suffix: view_cuda
     args: -ng 64 -kernel 1 -dim 2 -symm 1 -checkexpl -bgpu -B_view -mat_h2opus_leafsize 17 -mat_h2opus_normsamples 13 -mat_h2opus_indexmap_view ::ascii_matlab -mat_approximate_norm_samples 2 -mat_h2opus_normsamples 2

TEST*/
