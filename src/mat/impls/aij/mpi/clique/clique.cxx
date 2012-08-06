#include <../src/mat/impls/aij/mpi/clique/matcliqueimpl.h> /*I "petscmat.h" I*/
/*
 Provides an interface to the Clique sparse solver (http://poulson.github.com/Clique/)
*/

/*
  Convert Petsc aij matrix to Clique matrix C

  input:
    A       - matrix in seqaij or mpiaij format
    valOnly - FALSE: spaces are allocated and values are set for the Clique matrix C
              TRUE:  Only fill values
  output:
*/

#undef __FUNCT__
#define __FUNCT__ "MatConvertToClique"
PetscErrorCode MatConvertToClique(Mat A,PetscBool valOnly, cliq::DistSparseMatrix<PetscCliqScalar> **cmat)
{
  PetscErrorCode        ierr;
  PetscInt              i,j,rstart,rend,ncols;
  const PetscInt        *cols;
  const PetscCliqScalar *vals;
  cliq::mpi::Comm cxxcomm(((PetscObject)A)->comm);
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat_ptr;
  
  PetscFunctionBegin;
  printf("MatConvertToClique ...\n");
  if (!valOnly){ 
    printf("  create cmat...\n");
    /* create Clique matrix */
    cmat_ptr = new cliq::DistSparseMatrix<PetscCliqScalar>(A->rmap->N,cxxcomm);
    cmat = &cmat_ptr;
  } else {
    cmat_ptr = *cmat;
  }
  /* fill matrix values */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  const int firstLocalRow = cmat_ptr->FirstLocalRow();
  const int localHeight = cmat_ptr->LocalHeight();
  //printf("rstar,end: %d %d; firstLocalRow,localHeight: %d %d\n",rstart,rend,firstLocalRow,localHeight);
  if (rstart != firstLocalRow || rend-rstart != localHeight) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"matrix rowblock distribution does not match");

  cmat_ptr->StartAssembly();
  for (i=rstart; i<rend; i++){ 
    ierr = MatGetRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr); 
    //printf("row %d, ncols %d\n",i,ncols);
    for (j=0; j<ncols; j++){
      cmat_ptr->Update(i,cols[j],vals[j]);
    }
    ierr = MatRestoreRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr); 
  }
  cmat_ptr->StopAssembly();

  // Test cmat using Clique vectors
  PetscInt N=A->cmap->N;
  cliq::DistVector<double> xc1( N, cxxcomm), yc1( N, cxxcomm);
  cliq::MakeUniform( xc1 );
  const double xOrigNorm = cliq::Norm( xc1 );
  cliq::MakeZeros( yc1 );
  cliq::Multiply( 1., *cmat_ptr, xc1, 0., yc1 );
  const double yOrigNorm = cliq::Norm( yc1 );
  printf(" clique norm(xc1,yc1) %g %g\n",xOrigNorm,yOrigNorm);

  // Test cmat using petsc vectors - fail!
  Vec X,Y;
  i=0;
  PetscScalar zero=0.0,one=1.0;
  const PetscCliqScalar *x;
  PetscCliqScalar       *y;
 
  ierr = MatGetVecs(A,&X,&Y);CHKERRQ(ierr); 
  ierr = VecSet(X,zero);CHKERRQ(ierr);
  ierr = VecSetValues(X,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  printf("X:\n");
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);

  // must pass x to xc, y to yc!
  cliq::DistVector<PetscCliqScalar> xc(A->cmap->N,cxxcomm);
  cliq::DistVector<PetscCliqScalar> yc(A->rmap->N,cxxcomm);
  
  cliq::Multiply(1.0,*cmat_ptr,xc,0.0,yc);

  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  
  printf("Y = A*X:\n");
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Clique"
static PetscErrorCode MatMult_Clique(Mat A,Vec X,Vec Y)
{
  PetscErrorCode        ierr;
  PetscInt              i;
  const PetscCliqScalar *x;
  PetscCliqScalar       *y;
  Mat_Clique            *cliq=(Mat_Clique*)A->spptr;
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat=cliq->cmat;
  cliq::mpi::Comm cxxcomm(((PetscObject)A)->comm);

  PetscFunctionBegin;
  if (!cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Clique matrix cmat is not created yet");
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  //ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  //ierr = VecGetLocalSize(Y,&m);CHKERRQ(ierr);
  //ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  //ierr = VecGetSize(Y,&M);CHKERRQ(ierr);
  cliq::DistVector<PetscCliqScalar> xc(A->cmap->N,cxxcomm);
  cliq::DistVector<PetscCliqScalar> yc(A->rmap->N,cxxcomm);
  for (i=0; i<A->cmap->n; i++) {
    xc.SetLocal(i,x[i]);
  }
  cliq::Multiply(1.0,*cmat,xc,0.0,yc);
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_Clique"
PetscErrorCode MatView_Clique(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Info viewer not implemented yet");
    } else if (format == PETSC_VIEWER_DEFAULT) {
      Mat Aaij;
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscObjectPrintClassNamePrefixType((PetscObject)A,viewer,"Matrix Object");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscPrintf(((PetscObject)viewer)->comm,"Clique matrix\n");CHKERRQ(ierr);
      ierr = MatComputeExplicitOperator(A,&Aaij);CHKERRQ(ierr);
      ierr = MatView(Aaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = MatDestroy(&Aaij);CHKERRQ(ierr);     
    } else SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Format");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by Elemental matrices",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Clique"
PetscErrorCode MatDestroy_Clique(Mat A)
{
  PetscErrorCode ierr;
  Mat_Clique     *cliq=(Mat_Clique*)A->spptr;

  PetscFunctionBegin;
  printf("MatDestroy_Clique ...\n");
  if (cliq && cliq->CleanUpClique) { 
    /* Terminate instance, deallocate memories */
    printf("MatDestroy_Clique ... destroy clique struct \n");
    // free cmat here 
  }
  if (cliq && cliq->Destroy) {
    ierr = cliq->Destroy(A);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatFactorGetSolverPackage_C","",PETSC_NULL);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_Clique"
PetscErrorCode MatSolve_Clique(Mat A,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_Clique"
PetscErrorCode MatCholeskyFactorNumeric_Clique(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode    ierr;
  Mat_Clique        *cliq=(Mat_Clique*)F->spptr;
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat;

  PetscFunctionBegin;
  printf("MatCholeskyFactorNumeric_Clique \n");
  if (cliq->matstruc == SAME_NONZERO_PATTERN){ /* successing numerical factorization */
    /* Update cmat */
    ierr = MatConvertToClique(A,PETSC_TRUE,&cmat);CHKERRQ(ierr);
  }

  /* Numeric factorization */

  cliq->matstruc = SAME_NONZERO_PATTERN;
  F->assembled   = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_Clique"
PetscErrorCode MatCholeskyFactorSymbolic_Clique(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode    ierr;
  Mat_Clique        *cliq=(Mat_Clique*)F->spptr;
  cliq::DistSparseMatrix<PetscScalar> *cmat;

  PetscFunctionBegin;
  printf("MatCholeskyFactorSymbolic_Clique \n");
  /* Convert A to Aclique */
  ierr = MatConvertToClique(A,PETSC_FALSE,&cmat);CHKERRQ(ierr);
  cliq->cmat = cmat;

  cliq->matstruc      = DIFFERENT_NONZERO_PATTERN;
  cliq->CleanUpClique = PETSC_TRUE;

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_Clique"
PetscErrorCode MatFactorGetSolverPackage_Clique(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCLIQUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_aij_clique"
PetscErrorCode MatGetFactor_aij_clique(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_Clique     *cliq;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_CHOLESKY){
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_Clique;
    B->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_Clique;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");

  ierr = PetscNewLog(B,Mat_Clique,&cliq);CHKERRQ(ierr);
  B->spptr            = (void*)cliq;
  cliq->CleanUpClique = PETSC_FALSE;
  cliq->Destroy       = B->ops->destroy;

  B->ops->view    = MatView_Clique;
  B->ops->mult    = MatMult_Clique; //???
  B->ops->solve   = MatSolve_Clique;

  B->ops->destroy = MatDestroy_Clique;
  B->factortype   = ftype;
  B->assembled    = PETSC_FALSE;  /* required by -ksp_view */

  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END
