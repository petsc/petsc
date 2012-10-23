#include <../src/mat/impls/aij/mpi/clique/matcliqueimpl.h> /*I "petscmat.h" I*/
/*
 Provides an interface to the Clique sparse solver (http://poulson.github.com/Clique/)
*/

#undef __FUNCT__
#define __FUNCT__ "PetscCliqueFinalizePackage"
PetscErrorCode PetscCliqueFinalizePackage(void)
{
  PetscFunctionBegin;
  cliq::Finalize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCliqueInitializePackage"
PetscErrorCode PetscCliqueInitializePackage(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cliq::Initialized()) PetscFunctionReturn(0);
  { /* We have already initialized MPI, so this song and dance is just to pass these variables (which won't be used by Clique) through the interface that needs references */
    int zero = 0;
    char **nothing = 0;
    cliq::Initialize(zero,nothing);
  }
  ierr = PetscRegisterFinalize(PetscCliqueFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  MatConvertToClique: Convert Petsc aij matrix to Clique matrix 

  input:
+   A     - matrix in seqaij or mpiaij format
-   reuse - denotes if the destination matrix is to be created or reused. Currently
            MAT_REUSE_MATRIX is only supported for inplace conversion, otherwise use MAT_INITIAL_MATRIX.

  output:   
.   cliq - Clique context 
*/
#undef __FUNCT__
#define __FUNCT__ "MatConvertToClique"
PetscErrorCode MatConvertToClique(Mat A,MatReuse reuse,Mat_Clique *cliq)
{
  PetscErrorCode        ierr;
  PetscInt              i,j,rstart,rend,ncols;
  const PetscInt        *cols;
  const PetscCliqScalar *vals;
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat;
  
  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX){ 
    /* create Clique matrix */
    cmat = new cliq::DistSparseMatrix<PetscCliqScalar>(A->rmap->N,cliq->cliq_comm);
    cliq->cmat = cmat;
  } else {
    cmat = cliq->cmat;
  }
  /* fill matrix values */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  const int firstLocalRow = cmat->FirstLocalRow();
  const int localHeight = cmat->LocalHeight();
  if (rstart != firstLocalRow || rend-rstart != localHeight) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"matrix rowblock distribution does not match");

  cmat->StartAssembly();
  //cmat->Reserve( 7*localHeight ); ???
  for (i=rstart; i<rend; i++){ 
    ierr = MatGetRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr); 
    for (j=0; j<ncols; j++){
      cmat->Update(i,cols[j],vals[j]);
    }
    ierr = MatRestoreRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr); 
  }
  cmat->StopAssembly();
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
  
  cliq::DistVector<PetscCliqScalar> xc(A->cmap->N,cxxcomm);
  cliq::DistVector<PetscCliqScalar> yc(A->rmap->N,cxxcomm);
  for (i=0; i<A->cmap->n; i++) {
    xc.SetLocal(i,x[i]);
  }
  cliq::Multiply(1.0,*cmat,xc,0.0,yc);
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
 
  for (i=0; i<A->cmap->n; i++) {
    ierr = VecSetValueLocal(Y,i,yc.GetLocal(i),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);
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
      ierr = PetscViewerASCIIPrintf(viewer,"Clique run parameters:\n");CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_DEFAULT) { /* matrix A is factored matrix, remove this block */
      Mat Aaij;
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscObjectPrintClassNamePrefixType((PetscObject)A,viewer,"Matrix Object");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscPrintf(((PetscObject)viewer)->comm,"Clique matrix\n");CHKERRQ(ierr);
      ierr = MatComputeExplicitOperator(A,&Aaij);CHKERRQ(ierr);
      ierr = MatView(Aaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = MatDestroy(&Aaij);CHKERRQ(ierr);     
    } else SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Format");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by Clique matrices",((PetscObject)viewer)->type_name);
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
    ierr = PetscCommDestroy(&(cliq->cliq_comm));CHKERRQ(ierr);
    // free cmat here
    delete cliq->cmat;
    delete cliq->frontTree;
    delete cliq->rhs;
    delete cliq->xNodal;
    delete cliq->info;
    delete cliq->inverseMap;
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
PetscErrorCode MatSolve_Clique(Mat A,Vec B,Vec X)
{
  PetscErrorCode        ierr;
  PetscInt              i,rank;
  const PetscCliqScalar *b;
  Mat_Clique            *cliq=(Mat_Clique*)A->spptr;
  cliq::DistVector<PetscCliqScalar> *bc=cliq->rhs;
  cliq::DistNodalVector<PetscCliqScalar> *xNodal=cliq->xNodal;
 
  PetscFunctionBegin;
  ierr = VecGetArrayRead(B,(const PetscScalar **)&b);CHKERRQ(ierr);
  for (i=0; i<A->rmap->n; i++) {
    bc->SetLocal(i,b[i]);
  }
  ierr = VecRestoreArrayRead(B,(const PetscScalar **)&b);CHKERRQ(ierr);

  xNodal->Pull( *cliq->inverseMap, *cliq->info, *bc );
  cliq::Solve( *cliq->info, *cliq->frontTree, xNodal->localVec );
  xNodal->Push( *cliq->inverseMap, *cliq->info, *bc );

  ierr = MPI_Comm_rank(cliq->cliq_comm,&rank);CHKERRQ(ierr); 
  for (i=0; i<bc->LocalHeight(); i++) {
    ierr = VecSetValue(X,rank*bc->Blocksize()+i,bc->GetLocal(i),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
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
  cmat = cliq->cmat;
  if (cliq->matstruc == SAME_NONZERO_PATTERN){ /* successing numerical factorization */
    /* Update cmat */
    ierr = MatConvertToClique(A,MAT_REUSE_MATRIX,cliq);CHKERRQ(ierr);
  }

  /* Numeric factorization */
  cliq::LDL( *cliq->info, *cliq->frontTree, cliq::LDL_1D);
  //L.frontType = cliq::SYMM_2D;

  // refactor
  //cliq::ChangeFrontType( *cliq->frontTree, cliq::LDL_2D );
  //*(cliq->frontTree.frontType) = cliq::LDL_2D;
  //cliq::LDL( *cliq->info, *cliq->frontTree, cliq::LDL_2D );

  cliq->matstruc = SAME_NONZERO_PATTERN;
  F->assembled   = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_Clique"
PetscErrorCode MatCholeskyFactorSymbolic_Clique(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode    ierr;
  Mat_Clique        *Acliq=(Mat_Clique*)F->spptr;
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat;
  cliq::DistSeparatorTree                 sepTree;
  cliq::DistMap                           map;

  PetscFunctionBegin;
  /* Convert A to Aclique */
  ierr = MatConvertToClique(A,MAT_INITIAL_MATRIX,Acliq);CHKERRQ(ierr);
  cmat = Acliq->cmat;

  cliq::NestedDissection( cmat->Graph(), map, sepTree, *Acliq->info, PETSC_TRUE, Acliq->numDistSeps, Acliq->numSeqSeps, Acliq->cutoff);
  map.FormInverse( *Acliq->inverseMap );
  Acliq->frontTree = new cliq::DistSymmFrontTree<PetscCliqScalar>( cliq::TRANSPOSE, *cmat, map, sepTree, *Acliq->info );

  Acliq->matstruc      = DIFFERENT_NONZERO_PATTERN;
  Acliq->CleanUpClique = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*MC
     MATSOLVERCLIQUE  - A solver package providing direct solvers for distributed
  and sequential matrices via the external package Clique.

  Use ./configure --download-clique to have PETSc installed with Clique

  Options Database Keys:
+ -mat_clique_    - 
- -mat_clique_ <integer> - 

  Level: beginner

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/

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
  ierr = PetscCliqueInitializePackage(PETSC_NULL);CHKERRQ(ierr);
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
  cliq::mpi::Comm cxxcomm(((PetscObject)A)->comm);
  ierr = PetscCommDuplicate(cxxcomm,&(cliq->cliq_comm),PETSC_NULL);CHKERRQ(ierr);
  cliq->rhs           = new cliq::DistVector<PetscCliqScalar>(A->rmap->N,cliq->cliq_comm);
  cliq->xNodal        = new cliq::DistNodalVector<PetscCliqScalar>();
  cliq->info          = new cliq::DistSymmInfo;
  cliq->inverseMap    = new cliq::DistMap;
  cliq->CleanUpClique = PETSC_FALSE;
  cliq->Destroy       = B->ops->destroy;

  B->ops->view    = MatView_Clique;
  B->ops->mult    = MatMult_Clique; /* for cliq->cmat */
  B->ops->solve   = MatSolve_Clique;

  B->ops->destroy = MatDestroy_Clique;
  B->factortype   = ftype;
  B->assembled    = PETSC_FALSE;  

  /* Set Clique options */
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"Clique Options","Mat");CHKERRQ(ierr);
  cliq->cutoff      = 128;  /* maximum size of leaf node */
  cliq->numDistSeps = 1;    /* number of distributed separators to try */
  cliq->numSeqSeps  = 1;    /* number of sequential separators to try */
  PetscOptionsEnd();

  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END
