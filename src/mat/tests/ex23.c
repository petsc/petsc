
static char help[] = "Tests the use of interface functions for MATIS matrices.\n\
This example tests: MatZeroRows(), MatZeroRowsLocal(), MatView(), MatDuplicate(),\n\
MatCopy(), MatCreateSubMatrix(), MatGetLocalSubMatrix(), MatAXPY(), MatShift()\n\
MatDiagonalSet(), MatTranspose() and MatPtAP(). It also tests some\n\
conversion routines.\n";

#include <petscmat.h>

PetscErrorCode TestMatZeroRows(Mat,Mat,PetscBool,IS,PetscScalar);
PetscErrorCode CheckMat(Mat,Mat,PetscBool,const char*);

int main(int argc,char **args)
{
  Mat                    A,B,A2,B2,T;
  Mat                    Aee,Aeo,Aoe,Aoo;
  Mat                    *mats;
  Vec                    x,y;
  MatInfo                info;
  ISLocalToGlobalMapping cmap,rmap;
  IS                     is,is2,reven,rodd,ceven,codd;
  IS                     *rows,*cols;
  MatType                lmtype;
  PetscScalar            diag = 2.;
  PetscInt               n,m,i,lm,ln;
  PetscInt               rst,ren,cst,cen,nr,nc;
  PetscMPIInt            rank,size;
  PetscBool              testT,squaretest,isaij;
  PetscBool              permute = PETSC_FALSE, negmap = PETSC_FALSE, repmap = PETSC_FALSE;
  PetscBool              diffmap = PETSC_TRUE, symmetric = PETSC_FALSE, issymmetric;
  PetscErrorCode         ierr;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  m = n = 2*size;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-symmetric",&symmetric,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-negmap",&negmap,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-repmap",&repmap,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-permmap",&permute,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-diffmap",&diffmap,NULL));
  PetscCheckFalse(size > 1 && m < 4,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of rows should be larger or equal 4 for parallel runs");
  PetscCheckFalse(size == 1 && m < 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of rows should be larger or equal 2 for uniprocessor runs");
  PetscCheckFalse(n < 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of cols should be larger or equal 2");

  /* create a MATIS matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetType(A,MATIS));
  PetscCall(MatSetFromOptions(A));
  if (!negmap && !repmap) {
    /* This is not the proper setting for MATIS for finite elements, it is just used to test the routines
       Here we use a one-to-one correspondence between local row/column spaces and global row/column spaces
       Equivalent to passing NULL for the mapping */
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,0,1,&is));
  } else if (negmap && !repmap) { /* non repeated but with negative indices */
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,n+2,-2,1,&is));
  } else if (!negmap && repmap) { /* non negative but repeated indices */
    IS isl[2];

    PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,0,1,&isl[0]));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,n-1,-1,&isl[1]));
    PetscCall(ISConcatenate(PETSC_COMM_WORLD,2,isl,&is));
    PetscCall(ISDestroy(&isl[0]));
    PetscCall(ISDestroy(&isl[1]));
  } else { /* negative and repeated indices */
    IS isl[2];

    PetscCall(ISCreateStride(PETSC_COMM_WORLD,n+1,-1,1,&isl[0]));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,n+1,n-1,-1,&isl[1]));
    PetscCall(ISConcatenate(PETSC_COMM_WORLD,2,isl,&is));
    PetscCall(ISDestroy(&isl[0]));
    PetscCall(ISDestroy(&isl[1]));
  }
  PetscCall(ISLocalToGlobalMappingCreateIS(is,&cmap));
  PetscCall(ISDestroy(&is));

  if (m != n || diffmap) {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,m,permute ? m-1 : 0,permute ? -1 : 1,&is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is,&rmap));
    PetscCall(ISDestroy(&is));
  } else {
    PetscCall(PetscObjectReference((PetscObject)cmap));
    rmap = cmap;
  }

  PetscCall(MatSetLocalToGlobalMapping(A,rmap,cmap));
  PetscCall(MatISStoreL2L(A,PETSC_FALSE));
  PetscCall(MatISSetPreallocation(A,3,NULL,3,NULL));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,(PetscBool)!(repmap || negmap))); /* I do not want to precompute the pattern */
  PetscCall(ISLocalToGlobalMappingGetSize(rmap,&lm));
  PetscCall(ISLocalToGlobalMappingGetSize(cmap,&ln));
  for (i=0; i<lm; i++) {
    PetscScalar v[3];
    PetscInt    cols[3];

    cols[0] = (i-1+n)%n;
    cols[1] = i%n;
    cols[2] = (i+1)%n;
    v[0] = -1.*(symmetric ? PetscMin(i+1,cols[0]+1) : i+1);
    v[1] =  2.*(symmetric ? PetscMin(i+1,cols[1]+1) : i+1);
    v[2] = -1.*(symmetric ? PetscMin(i+1,cols[2]+1) : i+1);
    PetscCall(ISGlobalToLocalMappingApply(cmap,IS_GTOLM_MASK,3,cols,NULL,cols));
    PetscCall(MatSetValuesLocal(A,1,&i,3,cols,v,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* activate tests for square matrices with same maps only */
  PetscCall(MatHasCongruentLayouts(A,&squaretest));
  if (squaretest && rmap != cmap) {
    PetscInt nr, nc;

    PetscCall(ISLocalToGlobalMappingGetSize(rmap,&nr));
    PetscCall(ISLocalToGlobalMappingGetSize(cmap,&nc));
    if (nr != nc) squaretest = PETSC_FALSE;
    else {
      const PetscInt *idxs1,*idxs2;

      PetscCall(ISLocalToGlobalMappingGetIndices(rmap,&idxs1));
      PetscCall(ISLocalToGlobalMappingGetIndices(cmap,&idxs2));
      PetscCall(PetscArraycmp(idxs1,idxs2,nr,&squaretest));
      PetscCall(ISLocalToGlobalMappingRestoreIndices(rmap,&idxs1));
      PetscCall(ISLocalToGlobalMappingRestoreIndices(cmap,&idxs2));
    }
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE,&squaretest,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
  }

  /* test MatISGetLocalMat */
  PetscCall(MatISGetLocalMat(A,&B));
  PetscCall(MatGetType(B,&lmtype));

  /* test MatGetInfo */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatGetInfo\n"));
  PetscCall(MatGetInfo(A,MAT_LOCAL,&info));
  PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"Process  %2d: %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",PetscGlobalRank,(PetscInt)info.nz_used,
                                            (PetscInt)info.nz_allocated,(PetscInt)info.nz_unneeded,(PetscInt)info.assemblies,(PetscInt)info.mallocs);PetscCall(ierr);
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatGetInfo(A,MAT_GLOBAL_MAX,&info));
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"GlobalMax  : %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",(PetscInt)info.nz_used,
                                (PetscInt)info.nz_allocated,(PetscInt)info.nz_unneeded,(PetscInt)info.assemblies,(PetscInt)info.mallocs);PetscCall(ierr);
  PetscCall(MatGetInfo(A,MAT_GLOBAL_SUM,&info));
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"GlobalSum  : %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",(PetscInt)info.nz_used,
                                (PetscInt)info.nz_allocated,(PetscInt)info.nz_unneeded,(PetscInt)info.assemblies,(PetscInt)info.mallocs);PetscCall(ierr);

  /* test MatIsSymmetric */
  PetscCall(MatIsSymmetric(A,0.0,&issymmetric));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatIsSymmetric: %d\n",issymmetric));

  /* Create a MPIAIJ matrix, same as A */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetType(B,MATAIJ));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  PetscCall(MatSetLocalToGlobalMapping(B,rmap,cmap));
  PetscCall(MatMPIAIJSetPreallocation(B,3,NULL,3,NULL));
  PetscCall(MatMPIBAIJSetPreallocation(B,1,3,NULL,3,NULL));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(MatHYPRESetPreallocation(B,3,NULL,3,NULL));
#endif
  PetscCall(MatISSetPreallocation(B,3,NULL,3,NULL));
  PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,(PetscBool)!(repmap || negmap))); /* I do not want to precompute the pattern */
  for (i=0; i<lm; i++) {
    PetscScalar v[3];
    PetscInt    cols[3];

    cols[0] = (i-1+n)%n;
    cols[1] = i%n;
    cols[2] = (i+1)%n;
    v[0] = -1.*(symmetric ? PetscMin(i+1,cols[0]+1) : i+1);
    v[1] =  2.*(symmetric ? PetscMin(i+1,cols[1]+1) : i+1);
    v[2] = -1.*(symmetric ? PetscMin(i+1,cols[2]+1) : i+1);
    PetscCall(ISGlobalToLocalMappingApply(cmap,IS_GTOLM_MASK,3,cols,NULL,cols));
    PetscCall(MatSetValuesLocal(B,1,&i,3,cols,v,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* test MatView */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatView\n"));
  PetscCall(MatView(A,NULL));
  PetscCall(MatView(B,NULL));

  /* test CheckMat */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test CheckMat\n"));
  PetscCall(CheckMat(A,B,PETSC_FALSE,"CheckMat"));

  /* test MatDuplicate and MatAXPY */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatDuplicate and MatAXPY\n"));
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));
  PetscCall(CheckMat(A,A2,PETSC_FALSE,"MatDuplicate and MatAXPY"));

  /* test MatConvert */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatConvert_IS_XAIJ\n"));
  PetscCall(MatConvert(A2,MATAIJ,MAT_INITIAL_MATRIX,&B2));
  PetscCall(CheckMat(B,B2,PETSC_TRUE,"MatConvert_IS_XAIJ MAT_INITIAL_MATRIX"));
  PetscCall(MatConvert(A2,MATAIJ,MAT_REUSE_MATRIX,&B2));
  PetscCall(CheckMat(B,B2,PETSC_TRUE,"MatConvert_IS_XAIJ MAT_REUSE_MATRIX"));
  PetscCall(MatConvert(A2,MATAIJ,MAT_INPLACE_MATRIX,&A2));
  PetscCall(CheckMat(B,A2,PETSC_TRUE,"MatConvert_IS_XAIJ MAT_INPLACE_MATRIX"));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&B2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatConvert_XAIJ_IS\n"));
  PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
  PetscCall(MatConvert(B2,MATIS,MAT_INITIAL_MATRIX,&A2));
  PetscCall(CheckMat(A,A2,PETSC_TRUE,"MatConvert_XAIJ_IS MAT_INITIAL_MATRIX"));
  PetscCall(MatConvert(B2,MATIS,MAT_REUSE_MATRIX,&A2));
  PetscCall(CheckMat(A,A2,PETSC_TRUE,"MatConvert_XAIJ_IS MAT_REUSE_MATRIX"));
  PetscCall(MatConvert(B2,MATIS,MAT_INPLACE_MATRIX,&B2));
  PetscCall(CheckMat(A,B2,PETSC_TRUE,"MatConvert_XAIJ_IS MAT_INPLACE_MATRIX"));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&B2));
  PetscCall(PetscStrcmp(lmtype,MATSEQAIJ,&isaij));
  if (size == 1 && isaij) { /* tests special code paths in MatConvert_IS_XAIJ */
    PetscInt               ri, ci, rr[3] = {0,1,0}, cr[4] = {1,2,0,1}, rk[3] = {0,2,1}, ck[4] = {1,0,3,2};
    ISLocalToGlobalMapping tcmap,trmap;

    for (ri = 0; ri < 2; ri++) {
      PetscInt *r;

      r = (PetscInt*)(ri == 0 ? rr : rk);
      for (ci = 0; ci < 2; ci++) {
        PetscInt *c,rb,cb;

        c = (PetscInt*)(ci == 0 ? cr : ck);
        for (rb = 1; rb < 4; rb++) {
          PetscCall(ISCreateBlock(PETSC_COMM_SELF,rb,3,r,PETSC_COPY_VALUES,&is));
          PetscCall(ISLocalToGlobalMappingCreateIS(is,&trmap));
          PetscCall(ISDestroy(&is));
          for (cb = 1; cb < 4; cb++) {
            Mat  T,lT,T2;
            char testname[256];

            PetscCall(PetscSNPrintf(testname,sizeof(testname),"MatConvert_IS_XAIJ special case (%" PetscInt_FMT " %" PetscInt_FMT ", bs %" PetscInt_FMT " %" PetscInt_FMT ")",ri,ci,rb,cb));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test %s\n",testname));

            PetscCall(ISCreateBlock(PETSC_COMM_SELF,cb,4,c,PETSC_COPY_VALUES,&is));
            PetscCall(ISLocalToGlobalMappingCreateIS(is,&tcmap));
            PetscCall(ISDestroy(&is));

            PetscCall(MatCreate(PETSC_COMM_SELF,&T));
            PetscCall(MatSetSizes(T,PETSC_DECIDE,PETSC_DECIDE,rb*3,cb*4));
            PetscCall(MatSetType(T,MATIS));
            PetscCall(MatSetLocalToGlobalMapping(T,trmap,tcmap));
            PetscCall(ISLocalToGlobalMappingDestroy(&tcmap));
            PetscCall(MatISGetLocalMat(T,&lT));
            PetscCall(MatSetType(lT,MATSEQAIJ));
            PetscCall(MatSeqAIJSetPreallocation(lT,cb*4,NULL));
            PetscCall(MatSetRandom(lT,NULL));
            PetscCall(MatConvert(lT,lmtype,MAT_INPLACE_MATRIX,&lT));
            PetscCall(MatISRestoreLocalMat(T,&lT));
            PetscCall(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));

            PetscCall(MatConvert(T,MATAIJ,MAT_INITIAL_MATRIX,&T2));
            PetscCall(CheckMat(T,T2,PETSC_TRUE,"MAT_INITIAL_MATRIX"));
            PetscCall(MatConvert(T,MATAIJ,MAT_REUSE_MATRIX,&T2));
            PetscCall(CheckMat(T,T2,PETSC_TRUE,"MAT_REUSE_MATRIX"));
            PetscCall(MatDestroy(&T2));
            PetscCall(MatDuplicate(T,MAT_COPY_VALUES,&T2));
            PetscCall(MatConvert(T2,MATAIJ,MAT_INPLACE_MATRIX,&T2));
            PetscCall(CheckMat(T,T2,PETSC_TRUE,"MAT_INPLACE_MATRIX"));
            PetscCall(MatDestroy(&T));
            PetscCall(MatDestroy(&T2));
          }
          PetscCall(ISLocalToGlobalMappingDestroy(&trmap));
        }
      }
    }
  }

  /* test MatDiagonalScale */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatDiagonalScale\n"));
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));
  PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
  PetscCall(MatCreateVecs(A,&x,&y));
  PetscCall(VecSetRandom(x,NULL));
  if (issymmetric) {
    PetscCall(VecCopy(x,y));
  } else {
    PetscCall(VecSetRandom(y,NULL));
    PetscCall(VecScale(y,8.));
  }
  PetscCall(MatDiagonalScale(A2,y,x));
  PetscCall(MatDiagonalScale(B2,y,x));
  PetscCall(CheckMat(A2,B2,PETSC_FALSE,"MatDiagonalScale"));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&B2));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  /* test MatPtAP (A IS and B AIJ) */
  if (isaij && m == n) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatPtAP\n"));
    PetscCall(MatISStoreL2L(A,PETSC_TRUE));
    PetscCall(MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A2));
    PetscCall(MatPtAP(B,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B2));
    PetscCall(CheckMat(A2,B2,PETSC_FALSE,"MatPtAP MAT_INITIAL_MATRIX"));
    PetscCall(MatPtAP(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&A2));
    PetscCall(CheckMat(A2,B2,PETSC_FALSE,"MatPtAP MAT_REUSE_MATRIX"));
    PetscCall(MatDestroy(&A2));
    PetscCall(MatDestroy(&B2));
  }

  /* test MatGetLocalSubMatrix */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatGetLocalSubMatrix\n"));
  PetscCall(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&A2));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,lm/2+lm%2,0,2,&reven));
  PetscCall(ISComplement(reven,0,lm,&rodd));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,ln/2+ln%2,0,2,&ceven));
  PetscCall(ISComplement(ceven,0,ln,&codd));
  PetscCall(MatGetLocalSubMatrix(A2,reven,ceven,&Aee));
  PetscCall(MatGetLocalSubMatrix(A2,reven,codd,&Aeo));
  PetscCall(MatGetLocalSubMatrix(A2,rodd,ceven,&Aoe));
  PetscCall(MatGetLocalSubMatrix(A2,rodd,codd,&Aoo));
  for (i=0; i<lm; i++) {
    PetscInt    j,je,jo,colse[3], colso[3];
    PetscScalar ve[3], vo[3];
    PetscScalar v[3];
    PetscInt    cols[3];
    PetscInt    row = i/2;

    cols[0] = (i-1+n)%n;
    cols[1] = i%n;
    cols[2] = (i+1)%n;
    v[0] = -1.*(symmetric ? PetscMin(i+1,cols[0]+1) : i+1);
    v[1] =  2.*(symmetric ? PetscMin(i+1,cols[1]+1) : i+1);
    v[2] = -1.*(symmetric ? PetscMin(i+1,cols[2]+1) : i+1);
    PetscCall(ISGlobalToLocalMappingApply(cmap,IS_GTOLM_MASK,3,cols,NULL,cols));
    for (j=0,je=0,jo=0;j<3;j++) {
      if (cols[j]%2) {
        vo[jo] = v[j];
        colso[jo++] = cols[j]/2;
      } else {
        ve[je] = v[j];
        colse[je++] = cols[j]/2;
      }
    }
    if (i%2) {
      PetscCall(MatSetValuesLocal(Aoe,1,&row,je,colse,ve,ADD_VALUES));
      PetscCall(MatSetValuesLocal(Aoo,1,&row,jo,colso,vo,ADD_VALUES));
    } else {
      PetscCall(MatSetValuesLocal(Aee,1,&row,je,colse,ve,ADD_VALUES));
      PetscCall(MatSetValuesLocal(Aeo,1,&row,jo,colso,vo,ADD_VALUES));
    }
  }
  PetscCall(MatRestoreLocalSubMatrix(A2,reven,ceven,&Aee));
  PetscCall(MatRestoreLocalSubMatrix(A2,reven,codd,&Aeo));
  PetscCall(MatRestoreLocalSubMatrix(A2,rodd,ceven,&Aoe));
  PetscCall(MatRestoreLocalSubMatrix(A2,rodd,codd,&Aoo));
  PetscCall(ISDestroy(&reven));
  PetscCall(ISDestroy(&ceven));
  PetscCall(ISDestroy(&rodd));
  PetscCall(ISDestroy(&codd));
  PetscCall(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAXPY(A2,-1.,A,SAME_NONZERO_PATTERN));
  PetscCall(CheckMat(A2,NULL,PETSC_FALSE,"MatGetLocalSubMatrix"));
  PetscCall(MatDestroy(&A2));

  /* test MatConvert_Nest_IS */
  testT = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_trans",&testT,NULL));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatConvert_Nest_IS\n"));
  nr   = 2;
  nc   = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nr",&nr,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nc",&nc,NULL));
  if (testT) {
    PetscCall(MatGetOwnershipRange(A,&cst,&cen));
    PetscCall(MatGetOwnershipRangeColumn(A,&rst,&ren));
  } else {
    PetscCall(MatGetOwnershipRange(A,&rst,&ren));
    PetscCall(MatGetOwnershipRangeColumn(A,&cst,&cen));
  }
  PetscCall(PetscMalloc3(nr,&rows,nc,&cols,2*nr*nc,&mats));
  for (i=0;i<nr*nc;i++) {
    if (testT) {
      PetscCall(MatCreateTranspose(A,&mats[i]));
      PetscCall(MatTranspose(B,MAT_INITIAL_MATRIX,&mats[i+nr*nc]));
    } else {
      PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&mats[i]));
      PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&mats[i+nr*nc]));
    }
  }
  for (i=0;i<nr;i++) {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,ren-rst,i+rst,nr,&rows[i]));
  }
  for (i=0;i<nc;i++) {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,cen-cst,i+cst,nc,&cols[i]));
  }
  PetscCall(MatCreateNest(PETSC_COMM_WORLD,nr,rows,nc,cols,mats,&A2));
  PetscCall(MatCreateNest(PETSC_COMM_WORLD,nr,rows,nc,cols,mats+nr*nc,&B2));
  for (i=0;i<nr;i++) {
    PetscCall(ISDestroy(&rows[i]));
  }
  for (i=0;i<nc;i++) {
    PetscCall(ISDestroy(&cols[i]));
  }
  for (i=0;i<2*nr*nc;i++) {
    PetscCall(MatDestroy(&mats[i]));
  }
  PetscCall(PetscFree3(rows,cols,mats));
  PetscCall(MatConvert(B2,MATAIJ,MAT_INITIAL_MATRIX,&T));
  PetscCall(MatDestroy(&B2));
  PetscCall(MatConvert(A2,MATIS,MAT_INITIAL_MATRIX,&B2));
  PetscCall(CheckMat(B2,T,PETSC_TRUE,"MatConvert_Nest_IS MAT_INITIAL_MATRIX"));
  PetscCall(MatConvert(A2,MATIS,MAT_REUSE_MATRIX,&B2));
  PetscCall(CheckMat(B2,T,PETSC_TRUE,"MatConvert_Nest_IS MAT_REUSE_MATRIX"));
  PetscCall(MatDestroy(&B2));
  PetscCall(MatConvert(A2,MATIS,MAT_INPLACE_MATRIX,&A2));
  PetscCall(CheckMat(A2,T,PETSC_TRUE,"MatConvert_Nest_IS MAT_INPLACE_MATRIX"));
  PetscCall(MatDestroy(&T));
  PetscCall(MatDestroy(&A2));

  /* test MatCreateSubMatrix */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatCreateSubMatrix\n"));
  if (rank == 0) {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,1,1,1,&is));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,2,0,1,&is2));
  } else if (rank == 1) {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,1,0,1,&is));
    if (n > 3) {
      PetscCall(ISCreateStride(PETSC_COMM_WORLD,1,3,1,&is2));
    } else {
      PetscCall(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is2));
    }
  } else if (rank == 2 && n > 4) {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,n-4,4,1,&is2));
  } else {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is2));
  }
  PetscCall(MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&A2));
  PetscCall(MatCreateSubMatrix(B,is,is,MAT_INITIAL_MATRIX,&B2));
  PetscCall(CheckMat(A2,B2,PETSC_TRUE,"first MatCreateSubMatrix"));

  PetscCall(MatCreateSubMatrix(A,is,is,MAT_REUSE_MATRIX,&A2));
  PetscCall(MatCreateSubMatrix(B,is,is,MAT_REUSE_MATRIX,&B2));
  PetscCall(CheckMat(A2,B2,PETSC_FALSE,"reuse MatCreateSubMatrix"));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&B2));

  if (!issymmetric) {
    PetscCall(MatCreateSubMatrix(A,is,is2,MAT_INITIAL_MATRIX,&A2));
    PetscCall(MatCreateSubMatrix(B,is,is2,MAT_INITIAL_MATRIX,&B2));
    PetscCall(MatCreateSubMatrix(A,is,is2,MAT_REUSE_MATRIX,&A2));
    PetscCall(MatCreateSubMatrix(B,is,is2,MAT_REUSE_MATRIX,&B2));
    PetscCall(CheckMat(A2,B2,PETSC_FALSE,"second MatCreateSubMatrix"));
  }

  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&B2));
  PetscCall(ISDestroy(&is));
  PetscCall(ISDestroy(&is2));

  /* Create an IS required by MatZeroRows(): just rank zero provides the rows to be eliminated */
  if (size > 1) {
    if (rank == 0) {
      PetscInt st,len;

      st   = (m+1)/2;
      len  = PetscMin(m/2,PetscMax(m-(m+1)/2-1,0));
      PetscCall(ISCreateStride(PETSC_COMM_WORLD,len,st,1,&is));
    } else {
      PetscCall(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is));
    }
  } else {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,1,0,1,&is));
  }

  if (squaretest) { /* tests for square matrices only, with same maps for rows and columns */
    /* test MatDiagonalSet */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatDiagonalSet\n"));
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));
    PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    PetscCall(MatCreateVecs(A,NULL,&x));
    PetscCall(VecSetRandom(x,NULL));
    PetscCall(MatDiagonalSet(A2,x,INSERT_VALUES));
    PetscCall(MatDiagonalSet(B2,x,INSERT_VALUES));
    PetscCall(CheckMat(A2,B2,PETSC_FALSE,"MatDiagonalSet"));
    PetscCall(VecDestroy(&x));
    PetscCall(MatDestroy(&A2));
    PetscCall(MatDestroy(&B2));

    /* test MatShift (MatShift_IS internally uses MatDiagonalSet_IS with ADD_VALUES) */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatShift\n"));
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));
    PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    PetscCall(MatShift(A2,2.0));
    PetscCall(MatShift(B2,2.0));
    PetscCall(CheckMat(A2,B2,PETSC_FALSE,"MatShift"));
    PetscCall(MatDestroy(&A2));
    PetscCall(MatDestroy(&B2));

    /* nonzero diag value is supported for square matrices only */
    PetscCall(TestMatZeroRows(A,B,PETSC_TRUE,is,diag));
  }
  PetscCall(TestMatZeroRows(A,B,squaretest,is,0.0));
  PetscCall(ISDestroy(&is));

  /* test MatTranspose */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatTranspose\n"));
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&A2));
  PetscCall(MatTranspose(B,MAT_INITIAL_MATRIX,&B2));
  PetscCall(CheckMat(A2,B2,PETSC_FALSE,"initial matrix MatTranspose"));

  PetscCall(MatTranspose(A,MAT_REUSE_MATRIX,&A2));
  PetscCall(CheckMat(A2,B2,PETSC_FALSE,"reuse matrix (not in place) MatTranspose"));
  PetscCall(MatDestroy(&A2));

  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));
  PetscCall(MatTranspose(A2,MAT_INPLACE_MATRIX,&A2));
  PetscCall(CheckMat(A2,B2,PETSC_FALSE,"reuse matrix (in place) MatTranspose"));
  PetscCall(MatDestroy(&A2));

  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&A2));
  PetscCall(CheckMat(A2,B2,PETSC_FALSE,"reuse matrix (different type) MatTranspose"));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&B2));

  /* test MatISFixLocalEmpty */
  if (isaij) {
    PetscInt r[2];

    r[0] = 0;
    r[1] = PetscMin(m,n)-1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatISFixLocalEmpty\n"));
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));

    PetscCall(MatISFixLocalEmpty(A2,PETSC_TRUE));
    PetscCall(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
    PetscCall(CheckMat(A2,B,PETSC_FALSE,"MatISFixLocalEmpty (null)"));

    PetscCall(MatZeroRows(A2,2,r,0.0,NULL,NULL));
    PetscCall(MatViewFromOptions(A2,NULL,"-fixempty_view"));
    PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    PetscCall(MatZeroRows(B2,2,r,0.0,NULL,NULL));
    PetscCall(MatISFixLocalEmpty(A2,PETSC_TRUE));
    PetscCall(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(A2,NULL,"-fixempty_view"));
    PetscCall(CheckMat(A2,B2,PETSC_FALSE,"MatISFixLocalEmpty (rows)"));
    PetscCall(MatDestroy(&A2));

    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));
    PetscCall(MatZeroRows(A2,2,r,0.0,NULL,NULL));
    PetscCall(MatTranspose(A2,MAT_INPLACE_MATRIX,&A2));
    PetscCall(MatTranspose(B2,MAT_INPLACE_MATRIX,&B2));
    PetscCall(MatViewFromOptions(A2,NULL,"-fixempty_view"));
    PetscCall(MatISFixLocalEmpty(A2,PETSC_TRUE));
    PetscCall(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(A2,NULL,"-fixempty_view"));
    PetscCall(CheckMat(A2,B2,PETSC_FALSE,"MatISFixLocalEmpty (cols)"));

    PetscCall(MatDestroy(&A2));
    PetscCall(MatDestroy(&B2));

    if (squaretest) {
      PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));
      PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
      PetscCall(MatZeroRowsColumns(A2,2,r,0.0,NULL,NULL));
      PetscCall(MatZeroRowsColumns(B2,2,r,0.0,NULL,NULL));
      PetscCall(MatViewFromOptions(A2,NULL,"-fixempty_view"));
      PetscCall(MatISFixLocalEmpty(A2,PETSC_TRUE));
      PetscCall(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
      PetscCall(MatViewFromOptions(A2,NULL,"-fixempty_view"));
      PetscCall(CheckMat(A2,B2,PETSC_FALSE,"MatISFixLocalEmpty (rows+cols)"));
      PetscCall(MatDestroy(&A2));
      PetscCall(MatDestroy(&B2));
    }

  }

  /* test MatInvertBlockDiagonal
       special cases for block-diagonal matrices */
  if (m == n) {
    ISLocalToGlobalMapping map;
    Mat                    Abd,Bbd;
    IS                     is,bis;
    const PetscScalar      *isbd,*aijbd;
    PetscScalar            *vals;
    const PetscInt         *sts,*idxs;
    PetscInt               *idxs2,diff,perm,nl,bs,st,en,in;
    PetscBool              ok;

    for (diff = 0; diff < 3; diff++) {
      for (perm = 0; perm < 3; perm++) {
        for (bs = 1; bs < 4; bs++) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatInvertBlockDiagonal blockdiag %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",n,diff,perm,bs));
          PetscCall(PetscMalloc1(bs*bs,&vals));
          PetscCall(MatGetOwnershipRanges(A,&sts));
          switch (diff) {
          case 1: /* inverted layout by processes */
            in = 1;
            st = sts[size - rank - 1];
            en = sts[size - rank];
            nl = en - st;
            break;
          case 2: /* round-robin layout */
            in = size;
            st = rank;
            nl = n/size;
            if (rank < n%size) nl++;
            break;
          default: /* same layout */
            in = 1;
            st = sts[rank];
            en = sts[rank + 1];
            nl = en - st;
            break;
          }
          PetscCall(ISCreateStride(PETSC_COMM_WORLD,nl,st,in,&is));
          PetscCall(ISGetLocalSize(is,&nl));
          PetscCall(ISGetIndices(is,&idxs));
          PetscCall(PetscMalloc1(nl,&idxs2));
          for (i=0;i<nl;i++) {
            switch (perm) { /* invert some of the indices */
            case 2:
              idxs2[i] = rank%2 ? idxs[i] : idxs[nl-i-1];
              break;
            case 1:
              idxs2[i] = rank%2 ? idxs[nl-i-1] : idxs[i];
              break;
            default:
              idxs2[i] = idxs[i];
              break;
            }
          }
          PetscCall(ISRestoreIndices(is,&idxs));
          PetscCall(ISCreateBlock(PETSC_COMM_WORLD,bs,nl,idxs2,PETSC_OWN_POINTER,&bis));
          PetscCall(ISLocalToGlobalMappingCreateIS(bis,&map));
          PetscCall(MatCreateIS(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,bs*n,bs*n,map,map,&Abd));
          PetscCall(ISLocalToGlobalMappingDestroy(&map));
          PetscCall(MatISSetPreallocation(Abd,bs,NULL,0,NULL));
          for (i=0;i<nl;i++) {
            PetscInt b1,b2;

            for (b1=0;b1<bs;b1++) {
              for (b2=0;b2<bs;b2++) {
                vals[b1*bs + b2] = i*bs*bs + b1*bs + b2 + 1 + (b1 == b2 ? 1.0 : 0);
              }
            }
            PetscCall(MatSetValuesBlockedLocal(Abd,1,&i,1,&i,vals,INSERT_VALUES));
          }
          PetscCall(MatAssemblyBegin(Abd,MAT_FINAL_ASSEMBLY));
          PetscCall(MatAssemblyEnd(Abd,MAT_FINAL_ASSEMBLY));
          PetscCall(MatConvert(Abd,MATAIJ,MAT_INITIAL_MATRIX,&Bbd));
          PetscCall(MatInvertBlockDiagonal(Abd,&isbd));
          PetscCall(MatInvertBlockDiagonal(Bbd,&aijbd));
          PetscCall(MatGetLocalSize(Bbd,&nl,NULL));
          ok   = PETSC_TRUE;
          for (i=0;i<nl/bs;i++) {
            PetscInt b1,b2;

            for (b1=0;b1<bs;b1++) {
              for (b2=0;b2<bs;b2++) {
                if (PetscAbsScalar(isbd[i*bs*bs+b1*bs + b2]-aijbd[i*bs*bs+b1*bs + b2]) > PETSC_SMALL) ok = PETSC_FALSE;
                if (!ok) {
                  PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] ERROR block %" PetscInt_FMT ", entry %" PetscInt_FMT " %" PetscInt_FMT ": %g %g\n",rank,i,b1,b2,(double)PetscAbsScalar(isbd[i*bs*bs+b1*bs + b2]),(double)PetscAbsScalar(aijbd[i*bs*bs+b1*bs + b2])));
                  break;
                }
              }
              if (!ok) break;
            }
            if (!ok) break;
          }
          PetscCall(MatDestroy(&Abd));
          PetscCall(MatDestroy(&Bbd));
          PetscCall(PetscFree(vals));
          PetscCall(ISDestroy(&is));
          PetscCall(ISDestroy(&bis));
        }
      }
    }
  }
  /* free testing matrices */
  PetscCall(ISLocalToGlobalMappingDestroy(&cmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&rmap));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode CheckMat(Mat A, Mat B, PetscBool usemult, const char* func)
{
  Mat            Bcheck;
  PetscReal      error;

  PetscFunctionBeginUser;
  if (!usemult && B) {
    PetscBool hasnorm;

    PetscCall(MatHasOperation(B,MATOP_NORM,&hasnorm));
    if (!hasnorm) usemult = PETSC_TRUE;
  }
  if (!usemult) {
    if (B) {
      MatType Btype;

      PetscCall(MatGetType(B,&Btype));
      PetscCall(MatConvert(A,Btype,MAT_INITIAL_MATRIX,&Bcheck));
    } else {
      PetscCall(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&Bcheck));
    }
    if (B) { /* if B is present, subtract it */
      PetscCall(MatAXPY(Bcheck,-1.,B,DIFFERENT_NONZERO_PATTERN));
    }
    PetscCall(MatNorm(Bcheck,NORM_INFINITY,&error));
    if (error > PETSC_SQRT_MACHINE_EPSILON) {
      ISLocalToGlobalMapping rl2g,cl2g;

      PetscCall(PetscObjectSetName((PetscObject)Bcheck,"Assembled Bcheck"));
      PetscCall(MatView(Bcheck,NULL));
      if (B) {
        PetscCall(PetscObjectSetName((PetscObject)B,"Assembled AIJ"));
        PetscCall(MatView(B,NULL));
        PetscCall(MatDestroy(&Bcheck));
        PetscCall(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&Bcheck));
        PetscCall(PetscObjectSetName((PetscObject)Bcheck,"Assembled IS"));
        PetscCall(MatView(Bcheck,NULL));
      }
      PetscCall(MatDestroy(&Bcheck));
      PetscCall(PetscObjectSetName((PetscObject)A,"MatIS"));
      PetscCall(MatView(A,NULL));
      PetscCall(MatGetLocalToGlobalMapping(A,&rl2g,&cl2g));
      PetscCall(ISLocalToGlobalMappingView(rl2g,NULL));
      PetscCall(ISLocalToGlobalMappingView(cl2g,NULL));
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR ON %s: %g",func,(double)error);
    }
    PetscCall(MatDestroy(&Bcheck));
  } else {
    PetscBool ok,okt;

    PetscCall(MatMultEqual(A,B,3,&ok));
    PetscCall(MatMultTransposeEqual(A,B,3,&okt));
    PetscCheckFalse(!ok || !okt,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR ON %s: mult ok ?  %d, multtranspose ok ? %d",func,ok,okt);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TestMatZeroRows(Mat A, Mat Afull, PetscBool squaretest, IS is, PetscScalar diag)
{
  Mat                    B,Bcheck,B2 = NULL,lB;
  Vec                    x = NULL, b = NULL, b2 = NULL;
  ISLocalToGlobalMapping l2gr,l2gc;
  PetscReal              error;
  char                   diagstr[16];
  const PetscInt         *idxs;
  PetscInt               rst,ren,i,n,N,d;
  PetscMPIInt            rank;
  PetscBool              miss,haszerorows;

  PetscFunctionBeginUser;
  if (diag == 0.) {
    PetscCall(PetscStrcpy(diagstr,"zero"));
  } else {
    PetscCall(PetscStrcpy(diagstr,"nonzero"));
  }
  PetscCall(ISView(is,NULL));
  PetscCall(MatGetLocalToGlobalMapping(A,&l2gr,&l2gc));
  /* tests MatDuplicate and MatCopy */
  if (diag == 0.) {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  } else {
    PetscCall(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B));
    PetscCall(MatCopy(A,B,SAME_NONZERO_PATTERN));
  }
  PetscCall(MatISGetLocalMat(B,&lB));
  PetscCall(MatHasOperation(lB,MATOP_ZERO_ROWS,&haszerorows));
  if (squaretest && haszerorows) {

    PetscCall(MatCreateVecs(B,&x,&b));
    PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    PetscCall(VecSetLocalToGlobalMapping(b,l2gr));
    PetscCall(VecSetLocalToGlobalMapping(x,l2gc));
    PetscCall(VecSetRandom(x,NULL));
    PetscCall(VecSetRandom(b,NULL));
    /* mimic b[is] = x[is] */
    PetscCall(VecDuplicate(b,&b2));
    PetscCall(VecSetLocalToGlobalMapping(b2,l2gr));
    PetscCall(VecCopy(b,b2));
    PetscCall(ISGetLocalSize(is,&n));
    PetscCall(ISGetIndices(is,&idxs));
    PetscCall(VecGetSize(x,&N));
    for (i=0;i<n;i++) {
      if (0 <= idxs[i] && idxs[i] < N) {
        PetscCall(VecSetValue(b2,idxs[i],diag,INSERT_VALUES));
        PetscCall(VecSetValue(x,idxs[i],1.,INSERT_VALUES));
      }
    }
    PetscCall(VecAssemblyBegin(b2));
    PetscCall(VecAssemblyEnd(b2));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
    PetscCall(ISRestoreIndices(is,&idxs));
    /*  test ZeroRows on MATIS */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRows (diag %s)\n",diagstr));
    PetscCall(MatZeroRowsIS(B,is,diag,x,b));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRowsColumns (diag %s)\n",diagstr));
    PetscCall(MatZeroRowsColumnsIS(B2,is,diag,NULL,NULL));
  } else if (haszerorows) {
    /*  test ZeroRows on MATIS */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRows (diag %s)\n",diagstr));
    PetscCall(MatZeroRowsIS(B,is,diag,NULL,NULL));
    b = b2 = x = NULL;
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Skipping MatZeroRows (diag %s)\n",diagstr));
    b = b2 = x = NULL;
  }

  if (squaretest && haszerorows) {
    PetscCall(VecAXPY(b2,-1.,b));
    PetscCall(VecNorm(b2,NORM_INFINITY,&error));
    PetscCheckFalse(error > PETSC_SQRT_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR IN ZEROROWS ON B %g (diag %s)",(double)error,diagstr);
  }

  /* test MatMissingDiagonal */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test MatMissingDiagonal\n"));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(MatMissingDiagonal(B,&miss,&d));
  PetscCall(MatGetOwnershipRange(B,&rst,&ren));
  PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD, "[%d] [%" PetscInt_FMT ",%" PetscInt_FMT ") Missing %d, row %" PetscInt_FMT " (diag %s)\n",rank,rst,ren,(int)miss,d,diagstr));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&b2));

  /* check the result of ZeroRows with that from MPIAIJ routines
     assuming that MatConvert_IS_XAIJ and MatZeroRows_MPIAIJ work fine */
  if (haszerorows) {
    PetscCall(MatDuplicate(Afull,MAT_COPY_VALUES,&Bcheck));
    PetscCall(MatSetOption(Bcheck,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    PetscCall(MatZeroRowsIS(Bcheck,is,diag,NULL,NULL));
    PetscCall(CheckMat(B,Bcheck,PETSC_FALSE,"Zerorows"));
    PetscCall(MatDestroy(&Bcheck));
  }
  PetscCall(MatDestroy(&B));

  if (B2) { /* test MatZeroRowsColumns */
    PetscCall(MatDuplicate(Afull,MAT_COPY_VALUES,&B));
    PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    PetscCall(MatZeroRowsColumnsIS(B,is,diag,NULL,NULL));
    PetscCall(CheckMat(B2,B,PETSC_FALSE,"MatZeroRowsColumns"));
    PetscCall(MatDestroy(&B));
    PetscCall(MatDestroy(&B2));
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -test_trans

   test:
      suffix: 2
      nsize: 4
      args: -matis_convert_local_nest -nr 3 -nc 4

   test:
      suffix: 3
      nsize: 5
      args: -m 11 -n 10 -matis_convert_local_nest -nr 2 -nc 1

   test:
      suffix: 4
      nsize: 6
      args: -m 9 -n 12 -test_trans -nr 2 -nc 7

   test:
      suffix: 5
      nsize: 6
      args: -m 12 -n 12 -test_trans -nr 3 -nc 1

   test:
      suffix: 6
      args: -m 12 -n 12 -test_trans -nr 2 -nc 3 -diffmap

   test:
      suffix: 7
      args: -m 12 -n 12 -test_trans -nr 2 -nc 3 -diffmap -permmap

   test:
      suffix: 8
      args: -m 12 -n 17 -test_trans -nr 2 -nc 3 -permmap

   test:
      suffix: 9
      nsize: 5
      args: -m 12 -n 12 -test_trans -nr 2 -nc 3 -diffmap

   test:
      suffix: 10
      nsize: 5
      args: -m 12 -n 12 -test_trans -nr 2 -nc 3 -diffmap -permmap

   test:
      suffix: vscat_default
      nsize: 5
      args: -m 12 -n 17 -test_trans -nr 2 -nc 3 -permmap
      output_file: output/ex23_11.out

   test:
      suffix: 12
      nsize: 3
      args: -m 12 -n 12 -symmetric -matis_localmat_type sbaij -test_trans -nr 2 -nc 3

   testset:
      output_file: output/ex23_13.out
      nsize: 3
      args: -m 12 -n 17 -test_trans -nr 2 -nc 3 -diffmap -permmap
      filter: grep -v "type:"
      test:
        suffix: baij
        args: -matis_localmat_type baij
      test:
        requires: viennacl
        suffix: viennacl
        args: -matis_localmat_type aijviennacl
      test:
        requires: cuda
        suffix: cusparse
        args: -matis_localmat_type aijcusparse

   test:
      suffix: negrep
      nsize: {{1 3}separate output}
      args: -m {{5 7}separate output} -n {{5 7}separate output} -test_trans -nr 2 -nc 3 -negmap {{0 1}separate output} -repmap {{0 1}separate output} -permmap -diffmap {{0 1}separate output}

TEST*/
