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
  PetscInt       N, n = 64, dim = 1, i, j, nrhs = 11, lda = 0, ldc = 0, nt, ntrials = 2;
  PetscMPIInt    size,rank;
  PetscBool      testlayout = PETSC_FALSE,flg,symm = PETSC_FALSE, Asymm = PETSC_TRUE, kernel = PETSC_TRUE;
  PetscBool      checkexpl = PETSC_FALSE, agpu = PETSC_FALSE, bgpu = PETSC_FALSE, cgpu = PETSC_FALSE, flgglob;
  PetscBool      testtrans, testnorm, randommat = PETSC_TRUE, testorthog, testcompress;
  void           (*approxnormfunc)(void);
  void           (*Anormfunc)(void);
  PetscErrorCode ierr;

#if defined(PETSC_HAVE_MPI_INIT_THREAD)
  PETSC_MPI_THREAD_REQUIRED = MPI_THREAD_MULTIPLE;
#endif
  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-ng",&N,&flgglob);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-lda",&lda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ldc",&ldc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-matmattrials",&ntrials,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-randommat",&randommat,NULL);CHKERRQ(ierr);
  if (!flgglob) { ierr = PetscOptionsGetBool(NULL,NULL,"-testlayout",&testlayout,NULL);CHKERRQ(ierr); }
  ierr = PetscOptionsGetBool(NULL,NULL,"-Asymm",&Asymm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-symm",&symm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-kernel",&kernel,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-checkexpl",&checkexpl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-agpu",&agpu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-bgpu",&bgpu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-cgpu",&cgpu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-scale",&scale,NULL);CHKERRQ(ierr);
  if (!Asymm) symm = PETSC_FALSE;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  /* MatMultTranspose for nonsymmetric matrices in parallel not implemented */
  testtrans = (PetscBool)(size == 1 || symm);
  testnorm = (PetscBool)(size == 1 || symm);
  testorthog = (PetscBool)(size == 1 || symm);
  testcompress = (PetscBool)(size == 1 || symm);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscLayoutCreate(PETSC_COMM_WORLD,&map);CHKERRQ(ierr);
  if (testlayout) {
    if (rank%2) n = PetscMax(2*n-5*rank,0);
    else n = 2*n+rank;
  }
  if (!flgglob) {
    ierr = PetscLayoutSetLocalSize(map,n);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
    ierr = PetscLayoutGetSize(map,&N);CHKERRQ(ierr);
  } else {
    ierr = PetscLayoutSetSize(map,N);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
    ierr = PetscLayoutGetLocalSize(map,&n);CHKERRQ(ierr);
  }
  ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);

  if (lda) {
    ierr = PetscMalloc1(N*(n+lda),&Adata);CHKERRQ(ierr);
  }
  ierr = MatCreateDense(PETSC_COMM_WORLD,n,n,N,N,Adata,&A);CHKERRQ(ierr);
  ierr = MatDenseSetLDA(A,n+lda);CHKERRQ(ierr);

  /* Create random points; these are replicated in order to populate a dense matrix and to compare sequential and dense runs
     The constructor for MATH2OPUS can take as input the distributed coordinates and replicates them internally in case
     a kernel construction is requested */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(r,123456);CHKERRQ(ierr);
  ierr = PetscRandomSeed(r);CHKERRQ(ierr);
  ierr = PetscMalloc1(N*dim,&coords);CHKERRQ(ierr);
  ierr = PetscRandomGetValuesReal(r,N*dim,coords);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);

  if (kernel || !randommat) {
    MatH2OpusKernel k = Asymm ? GenEntry_Symm : GenEntry_Unsymm;
    PetscInt        ist,ien;

    ierr = MatGetOwnershipRange(A,&ist,&ien);CHKERRQ(ierr);
    for (i = ist; i < ien; i++) {
      for (j = 0; j < N; j++) {
        ierr = MatSetValue(A,i,j,(*k)(dim,coords + i*dim,coords + j*dim,NULL),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (kernel) {
      ierr = MatCreateH2OpusFromKernel(PETSC_COMM_WORLD,n,n,N,N,dim,coords + ist*dim,PETSC_TRUE,k,NULL,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&B);CHKERRQ(ierr);
    } else {
      ierr = MatCreateH2OpusFromMat(A,dim,coords + ist*dim,PETSC_TRUE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&B);CHKERRQ(ierr);
    }
  } else {
    PetscInt ist;

    ierr = MatGetOwnershipRange(A,&ist,NULL);CHKERRQ(ierr);
    ierr = MatSetRandom(A,NULL);CHKERRQ(ierr);
    if (Asymm) {
      ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      ierr = MatAXPY(A,1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&B);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = MatCreateH2OpusFromMat(A,dim,coords + ist*dim,PETSC_TRUE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&B);CHKERRQ(ierr);
  }
  ierr = PetscFree(coords);CHKERRQ(ierr);
  if (agpu) {
    ierr = MatConvert(A,MATDENSECUDA,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  }
  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr);

  ierr = MatSetOption(B,MAT_SYMMETRIC,symm);CHKERRQ(ierr);

  /* assemble the H-matrix */
  ierr = MatBindToCPU(B,(PetscBool)!bgpu);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(B,NULL,"-B_view");CHKERRQ(ierr);

  /* Test MatScale */
  ierr = MatScale(A,scale);CHKERRQ(ierr);
  ierr = MatScale(B,scale);CHKERRQ(ierr);

  /* Test MatMult */
  ierr = MatCreateVecs(A,&Ax,&Ay);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&Bx,&By);CHKERRQ(ierr);
  ierr = VecSetRandom(Ax,NULL);CHKERRQ(ierr);
  ierr = VecCopy(Ax,Bx);CHKERRQ(ierr);
  ierr = MatMult(A,Ax,Ay);CHKERRQ(ierr);
  ierr = MatMult(B,Bx,By);CHKERRQ(ierr);
  ierr = VecViewFromOptions(Ay,NULL,"-mult_vec_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(By,NULL,"-mult_vec_view");CHKERRQ(ierr);
  ierr = VecNorm(Ay,NORM_INFINITY,&nX);CHKERRQ(ierr);
  ierr = VecAXPY(Ay,-1.0,By);CHKERRQ(ierr);
  ierr = VecViewFromOptions(Ay,NULL,"-mult_vec_view");CHKERRQ(ierr);
  ierr = VecNorm(Ay,NORM_INFINITY,&err);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMult err %g\n",err/nX);CHKERRQ(ierr);
  ierr = VecScale(By,-1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(B,Bx,By,By);CHKERRQ(ierr);
  ierr = VecNorm(By,NORM_INFINITY,&err);CHKERRQ(ierr);
  ierr = VecViewFromOptions(By,NULL,"-mult_vec_view");CHKERRQ(ierr);
  if (err > 10.*PETSC_SMALL) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMultAdd err %g\n",err);CHKERRQ(ierr);
  }

  /* Test MatNorm */
  ierr = MatNorm(A,NORM_INFINITY,&norms[0]);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_1,&norms[1]);CHKERRQ(ierr);
  norms[2] = -1.; /* NORM_2 not supported */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A Matrix norms:        infty=%g, norm_1=%g, norm_2=%g\n",(double)norms[0],(double)norms[1],(double)norms[2]);CHKERRQ(ierr);
  ierr = MatGetOperation(A,MATOP_NORM,&Anormfunc);CHKERRQ(ierr);
  ierr = MatGetOperation(B,MATOP_NORM,&approxnormfunc);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_NORM,approxnormfunc);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_INFINITY,&norms[0]);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_1,&norms[1]);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_2,&norms[2]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A Approx Matrix norms: infty=%g, norm_1=%g, norm_2=%g\n",(double)norms[0],(double)norms[1],(double)norms[2]);CHKERRQ(ierr);
  if (testnorm) {
    ierr = MatNorm(B,NORM_INFINITY,&norms[0]);CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_1,&norms[1]);CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_2,&norms[2]);CHKERRQ(ierr);
  } else {
    norms[0] = -1.;
    norms[1] = -1.;
    norms[2] = -1.;
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"B Approx Matrix norms: infty=%g, norm_1=%g, norm_2=%g\n",(double)norms[0],(double)norms[1],(double)norms[2]);CHKERRQ(ierr);
  ierr = MatSetOperation(A,MATOP_NORM,Anormfunc);CHKERRQ(ierr);

  /* Test MatDuplicate */
  ierr = MatDuplicate(B,MAT_COPY_VALUES,&D);CHKERRQ(ierr);
  ierr = MatSetOption(D,MAT_SYMMETRIC,symm);CHKERRQ(ierr);
  ierr = MatMultEqual(B,D,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMult error after MatDuplicate\n");CHKERRQ(ierr);
  }
  if (testtrans) {
    ierr = MatMultTransposeEqual(B,D,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMultTranspose error after MatDuplicate\n");CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
    ierr = VecSetRandom(Ay,NULL);CHKERRQ(ierr);
    ierr = VecCopy(Ay,By);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,Ay,Ax);CHKERRQ(ierr);
    ierr = MatMultTranspose(B,By,Bx);CHKERRQ(ierr);
    ierr = VecViewFromOptions(Ax,NULL,"-multtrans_vec_view");CHKERRQ(ierr);
    ierr = VecViewFromOptions(Bx,NULL,"-multtrans_vec_view");CHKERRQ(ierr);
    ierr = VecNorm(Ax,NORM_INFINITY,&nX);CHKERRQ(ierr);
    ierr = VecAXPY(Ax,-1.0,Bx);CHKERRQ(ierr);
    ierr = VecViewFromOptions(Ax,NULL,"-multtrans_vec_view");CHKERRQ(ierr);
    ierr = VecNorm(Ax,NORM_INFINITY,&err);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMultTranspose err %g\n",err/nX);CHKERRQ(ierr);
    ierr = VecScale(Bx,-1.0);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(B,By,Bx,Bx);CHKERRQ(ierr);
    ierr = VecNorm(Bx,NORM_INFINITY,&err);CHKERRQ(ierr);
    if (err > 10.*PETSC_SMALL) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMultTransposeAdd err %g\n",err);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  ierr = VecDestroy(&Ay);CHKERRQ(ierr);
  ierr = VecDestroy(&Bx);CHKERRQ(ierr);
  ierr = VecDestroy(&By);CHKERRQ(ierr);

  /* Test MatMatMult */
  if (ldc) {
    ierr = PetscMalloc1(nrhs*(n+ldc),&Cdata);CHKERRQ(ierr);
  }
  ierr = MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,N,nrhs,Cdata,&C);CHKERRQ(ierr);
  ierr = MatDenseSetLDA(C,n+ldc);CHKERRQ(ierr);
  ierr = MatSetRandom(C,NULL);CHKERRQ(ierr);
  if (cgpu) {
    ierr = MatConvert(C,MATDENSECUDA,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
  }
  for (nt = 0; nt < ntrials; nt++) {
    ierr = MatMatMult(B,C,nt ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D);CHKERRQ(ierr);
    ierr = MatViewFromOptions(D,NULL,"-bc_view");CHKERRQ(ierr);
    ierr = PetscObjectBaseTypeCompareAny((PetscObject)D,&flg,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
    if (flg) {
      ierr = MatCreateVecs(B,&x,&y);CHKERRQ(ierr);
      ierr = MatCreateVecs(D,NULL,&v);CHKERRQ(ierr);
      for (i = 0; i < nrhs; i++) {
        ierr = MatGetColumnVector(D,v,i);CHKERRQ(ierr);
        ierr = MatGetColumnVector(C,x,i);CHKERRQ(ierr);
        ierr = MatMult(B,x,y);CHKERRQ(ierr);
        ierr = VecAXPY(y,-1.0,v);CHKERRQ(ierr);
        ierr = VecNorm(y,NORM_INFINITY,&err);CHKERRQ(ierr);
        if (err > 10.*PETSC_SMALL) { ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMat err %" PetscInt_FMT " %g\n",i,err);CHKERRQ(ierr); }
      }
      ierr = VecDestroy(&y);CHKERRQ(ierr);
      ierr = VecDestroy(&x);CHKERRQ(ierr);
      ierr = VecDestroy(&v);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* Test MatTransposeMatMult */
  if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
    for (nt = 0; nt < ntrials; nt++) {
      ierr = MatTransposeMatMult(B,C,nt ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D);CHKERRQ(ierr);
      ierr = MatViewFromOptions(D,NULL,"-btc_view");CHKERRQ(ierr);
      ierr = PetscObjectBaseTypeCompareAny((PetscObject)D,&flg,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
      if (flg) {
        ierr = MatCreateVecs(B,&y,&x);CHKERRQ(ierr);
        ierr = MatCreateVecs(D,NULL,&v);CHKERRQ(ierr);
        for (i = 0; i < nrhs; i++) {
          ierr = MatGetColumnVector(D,v,i);CHKERRQ(ierr);
          ierr = MatGetColumnVector(C,x,i);CHKERRQ(ierr);
          ierr = MatMultTranspose(B,x,y);CHKERRQ(ierr);
          ierr = VecAXPY(y,-1.0,v);CHKERRQ(ierr);
          ierr = VecNorm(y,NORM_INFINITY,&err);CHKERRQ(ierr);
          if (err > 10.*PETSC_SMALL) { ierr = PetscPrintf(PETSC_COMM_WORLD,"MatTransMat err %" PetscInt_FMT " %g\n",i,err);CHKERRQ(ierr); }
        }
        ierr = VecDestroy(&y);CHKERRQ(ierr);
        ierr = VecDestroy(&x);CHKERRQ(ierr);
        ierr = VecDestroy(&v);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&D);CHKERRQ(ierr);
  }

  if (testorthog) {
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&D);CHKERRQ(ierr);
    ierr = MatSetOption(D,MAT_SYMMETRIC,symm);CHKERRQ(ierr);
    ierr = MatH2OpusOrthogonalize(D);CHKERRQ(ierr);
    ierr = MatMultEqual(B,D,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMult error after basis ortogonalization\n");CHKERRQ(ierr);
    }
    ierr = MatDestroy(&D);CHKERRQ(ierr);
  }

  if (testcompress) {
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&D);CHKERRQ(ierr);
    ierr = MatSetOption(D,MAT_SYMMETRIC,symm);CHKERRQ(ierr);
    ierr = MatH2OpusCompress(D,PETSC_SMALL);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
  }

  /* check explicit operator */
  if (checkexpl) {
    Mat Be, Bet;

    ierr = MatComputeOperator(B,MATDENSE,&D);CHKERRQ(ierr);
    ierr = MatDuplicate(D,MAT_COPY_VALUES,&Be);CHKERRQ(ierr);
    ierr = MatNorm(D,NORM_FROBENIUS,&nB);CHKERRQ(ierr);
    ierr = MatViewFromOptions(D,NULL,"-expl_view");CHKERRQ(ierr);
    ierr = MatAXPY(D,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatViewFromOptions(D,NULL,"-diff_view");CHKERRQ(ierr);
    ierr = MatNorm(D,NORM_FROBENIUS,&nD);CHKERRQ(ierr);
    ierr = MatNorm(A,NORM_FROBENIUS,&nA);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Approximation error %g (%g / %g, %g)\n",nD/nA,nD,nA,nB);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);

    if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
      ierr = MatTranspose(A,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
      ierr = MatComputeOperatorTranspose(B,MATDENSE,&D);CHKERRQ(ierr);
      ierr = MatDuplicate(D,MAT_COPY_VALUES,&Bet);CHKERRQ(ierr);
      ierr = MatNorm(D,NORM_FROBENIUS,&nB);CHKERRQ(ierr);
      ierr = MatViewFromOptions(D,NULL,"-expl_trans_view");CHKERRQ(ierr);
      ierr = MatAXPY(D,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatViewFromOptions(D,NULL,"-diff_trans_view");CHKERRQ(ierr);
      ierr = MatNorm(D,NORM_FROBENIUS,&nD);CHKERRQ(ierr);
      ierr = MatNorm(A,NORM_FROBENIUS,&nA);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Approximation error transpose %g (%g / %g, %g)\n",nD/nA,nD,nA,nB);CHKERRQ(ierr);
      ierr = MatDestroy(&D);CHKERRQ(ierr);

      ierr = MatTranspose(Bet,MAT_INPLACE_MATRIX,&Bet);CHKERRQ(ierr);
      ierr = MatAXPY(Be,-1.0,Bet,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatViewFromOptions(Be,NULL,"-diff_expl_view");CHKERRQ(ierr);
      ierr = MatNorm(Be,NORM_FROBENIUS,&nB);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Approximation error B - (B^T)^T %g\n",nB);CHKERRQ(ierr);
      ierr = MatDestroy(&Be);CHKERRQ(ierr);
      ierr = MatDestroy(&Bet);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFree(Cdata);CHKERRQ(ierr);
  ierr = PetscFree(Adata);CHKERRQ(ierr);
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
     args: -n 33 -kernel 1 -dim 1 -lda 13 -ldc 11 -symm 0 -checkexpl -bgpu 0

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
     args: -n 33 -kernel 1 -dim 1 -lda 13 -ldc 11 -symm 0 -checkexpl -bgpu 1

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
