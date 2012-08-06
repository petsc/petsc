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
  PetscErrorCode    ierr;
  PetscInt          i,j,rstart,rend,ncols;
  const PetscInt    *cols;
  const PetscCliqScalar *vals;
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat_ptr;
 
  PetscFunctionBegin;
  printf("MatConvertToClique ...\n");
  if (!valOnly){ 
    /* create Clique matrix */
    cliq::mpi::Comm cxxcomm(((PetscObject)A)->comm);
    cmat_ptr = new cliq::DistSparseMatrix<PetscCliqScalar>(A->rmap->N,cxxcomm);
    cmat = &cmat_ptr;
  } else {
    cmat_ptr = *cmat;
  }
  /* fill matrix values */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  cmat_ptr->StartAssembly();
  for (i=rstart; i<rend; i++){
    ierr = MatGetRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr); 
    for (j=0; j<ncols; j++){
      cmat_ptr->Update(i,cols[j],vals[j]);
    }
    ierr = MatRestoreRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr); 
  }
  cmat_ptr->StopAssembly();
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
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat;
  cliq::mpi::Comm cxxcomm(((PetscObject)A)->comm);

  PetscFunctionBegin;
  ierr = MatConvertToClique(A,PETSC_FALSE,&cmat);CHKERRQ(ierr);
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
  //Mat_Elemental  *a = (Mat_Elemental*)A->data;
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
  PetscFunctionBegin;
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
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat;

  PetscFunctionBegin;
  /* Convert A to Aclique */
  ierr = MatConvertToClique(A,PETSC_FALSE,&cmat);CHKERRQ(ierr);

  /* Numeric factorization */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_Clique"
PetscErrorCode MatCholeskyFactorSymbolic_Clique(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode    ierr;
  cliq::DistSparseMatrix<PetscScalar> *cmat;

  PetscFunctionBegin;
  F->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_Clique;
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
  Mat_Clique     *Acliq;
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

  B->ops->destroy = MatDestroy_Clique;
  B->ops->view    = MatView_Clique;
  B->ops->mult    = MatMult_Clique;
  B->factortype   = ftype; 
  B->assembled    = PETSC_FALSE;  
  
  ierr = PetscNewLog(B,Mat_Clique,&Acliq);CHKERRQ(ierr);
  B->spptr = Acliq;
  *F       = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END
