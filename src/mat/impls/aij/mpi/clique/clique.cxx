#include <../src/mat/impls/aij/mpi/clique/matcliqueimpl.h> /*I "petscmat.h" I*/

/*
  MatConvertToSparseElemental: Convert Petsc aij matrix to sparse elemental matrix

  input:
+   A     - matrix in seqaij or mpiaij format
-   reuse - denotes if the destination matrix is to be created or reused. 
            Use MAT_INPLACE_MATRIX for inplace conversion, otherwise use MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX.

  output:
.   cliq - Clique context
*/
PetscErrorCode MatConvertToSparseElemental(Mat A,MatReuse reuse,Mat_SparseElemental *cliq)
{
  PetscErrorCode                          ierr;
  PetscInt                                i,j,rstart,rend,ncols;
  const PetscInt                          *cols;
  const PetscElemScalar                   *vals;
  El::DistSparseMatrix<PetscElemScalar> *cmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX){
    cmat = new El::DistSparseMatrix<PetscElemScalar>(A->rmap->N,cliq->comm);
    cliq->cmat = cmat;
  } else {
    cmat = cliq->cmat;
  }
  /* fill matrix values */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  const int firstLocalRow = cmat->FirstLocalRow();
  const int localHeight = cmat->LocalHeight();
  if (rstart != firstLocalRow || rend-rstart != localHeight) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"matrix rowblock distribution does not match");

  /* cmat->StartAssembly(); */
  //cmat->Reserve( 7*localHeight ); ???
  for (i=rstart; i<rend; i++){
    ierr = MatGetRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<ncols; j++){
      cmat->Update(i,cols[j],vals[j]);
    }
    ierr = MatRestoreRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  /* cmat->StopAssembly(); */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SparseElemental(Mat A,Vec X,Vec Y)
{
  PetscErrorCode                          ierr;
  PetscInt                                i;
  const PetscElemScalar                   *x;
  Mat_SparseElemental                              *cliq=(Mat_SparseElemental*)A->data;
  El::DistSparseMatrix<PetscElemScalar> *cmat=cliq->cmat;
  El::mpi::Comm                         cxxcomm(PetscObjectComm((PetscObject)A));

  PetscFunctionBegin;
  if (!cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Matrix cmat is not created yet");
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);

  El::DistMultiVec<PetscElemScalar> xc(A->cmap->N,1,cxxcomm);
  El::DistMultiVec<PetscElemScalar> yc(A->rmap->N,1,cxxcomm);
  for (i=0; i<A->cmap->n; i++) {
    xc.SetLocal(i,0,x[i]);
  }
  /* El::Multiply(1.0,*cmat,xc,0.0,yc); */
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);

  for (i=0; i<A->cmap->n; i++) {
    ierr = VecSetValueLocal(Y,i,yc.GetLocal(i,0),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SparseElemental(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"SparseElemental run parameters:\n");CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_DEFAULT) { /* matrix A is factored matrix, remove this block */
      Mat Aaij;
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)viewer),"SparseElemental matrix\n");CHKERRQ(ierr);
      ierr = MatComputeExplicitOperator(A,&Aaij);CHKERRQ(ierr);
      ierr = MatView(Aaij,viewer);CHKERRQ(ierr);
      ierr = MatDestroy(&Aaij);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SparseElemental(Mat A)
{
  PetscErrorCode ierr;
  Mat_SparseElemental     *cliq=(Mat_SparseElemental*)A->data;

  PetscFunctionBegin;
  if (cliq->CleanUp) {
    /* Terminate instance, deallocate memories */
    ierr = PetscCommDestroy(&(cliq->comm));CHKERRQ(ierr);
    // free cmat here
    delete cliq->cmat;
    delete cliq->rhs;
    delete cliq->inverseMap;
  }
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SparseElemental(Mat A,Vec B,Vec X)
{
  PetscErrorCode                     ierr;
  PetscInt                           i;
  PetscMPIInt                        rank;
  const PetscElemScalar              *b;
  Mat_SparseElemental                *cliq=(Mat_SparseElemental*)A->data;
  El::DistMultiVec<PetscElemScalar>  *bc=cliq->rhs;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(B,(const PetscScalar **)&b);CHKERRQ(ierr);
  for (i=0; i<A->rmap->n; i++) {
    bc->SetLocal(i,0,b[i]);
  }
  ierr = VecRestoreArrayRead(B,(const PetscScalar **)&b);CHKERRQ(ierr);

  /* xNodal->Pull( *cliq->inverseMap, *cliq->info, *bc ); */
  /* El::Solve( *cliq->info, *cliq->frontTree, *xNodal); */
  /* xNodal->Push( *cliq->inverseMap, *cliq->info, *bc ); */

  ierr = MPI_Comm_rank(cliq->comm,&rank);CHKERRQ(ierr);
  for (i=0; i<bc->LocalHeight(); i++) {
    ierr = VecSetValue(X,rank*bc->Blocksize()+i,bc->GetLocal(i,0),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorNumeric_SparseElemental(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode                          ierr;
  Mat_SparseElemental                              *cliq=(Mat_SparseElemental*)F->data;
  PETSC_UNUSED
  El::DistSparseMatrix<PetscElemScalar> *cmat;

  PetscFunctionBegin;
  cmat = cliq->cmat;
  if (cliq->matstruc == SAME_NONZERO_PATTERN){ /* successing numerical factorization */
    /* Update cmat */
    ierr = MatConvertToSparseElemental(A,MAT_REUSE_MATRIX,cliq);CHKERRQ(ierr);
  }

  /* Numeric factorization */
  /* El::LDL( *cliq->info, *cliq->frontTree, El::LDL_1D); */
  //L.frontType = El::SYMM_2D;

  // refactor
  //El::ChangeFrontType( *cliq->frontTree, El::LDL_2D );
  //*(cliq->frontTree.frontType) = El::LDL_2D;
  //El::LDL( *cliq->info, *cliq->frontTree, El::LDL_2D );

  cliq->matstruc = SAME_NONZERO_PATTERN;
  F->assembled   = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorSymbolic_SparseElemental(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode                          ierr;
  Mat_SparseElemental                              *Acliq=(Mat_SparseElemental*)F->data;
  El::DistSparseMatrix<PetscElemScalar> *cmat;
  El::DistMap                           map;

  PetscFunctionBegin;
  ierr = MatConvertToSparseElemental(A,MAT_INITIAL_MATRIX,Acliq);CHKERRQ(ierr);
  cmat = Acliq->cmat;

  /* El::NestedDissection( cmat->DistGraph(), map, sepTree, *Acliq->info, PETSC_TRUE, Acliq->numDistSeps, Acliq->numSeqSeps, Acliq->cutoff); */
  /* map.FormInverse( *Acliq->inverseMap ); */
  /* Acliq->frontTree = new El::DistSymmFrontTree<PetscElemScalar>( El::TRANSPOSE, *cmat, map, sepTree, *Acliq->info );*/

  Acliq->matstruc = DIFFERENT_NONZERO_PATTERN;
  Acliq->CleanUp  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*MC
     MATSOLVERSPARSEELEMENTAL  - A solver package providing direct solvers for sparse distributed
  and sequential matrices via the external package Elemental

  Use ./configure --download-elemental to have PETSc installed with Elemental

  Use -pc_type lu -pc_factor_mat_solver_package sparseelemental to us this direct solver

  This is currently not supported.

  Developer Note: Jed Brown made the interface for Clique when it was a standalone package. Later Jack Poulson merged and refactored Clique into
  Elemental but since the Clique interface was not tested in PETSc the interface was not updated for the new Elemental interface. Later Barry Smith updated
  all the boilerplate for the Clique interface to SparseElemental but since the solver interface changed dramatically he did not update the code
  that actually calls the SparseElemental solvers. We are waiting on someone who has a need to complete the SparseElemental interface from PETSc.

  Level: beginner

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/

PetscErrorCode MatFactorGetSolverPackage_SparseElemental(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSPARSEELEMENTAL;
  PetscFunctionReturn(0);
}

extern PetscErrorCode PetscElementalInitializePackage(void);

static PetscErrorCode MatGetFactor_aij_sparseelemental(Mat A,MatFactorType ftype,Mat *F)
{
  Mat                 B;
  Mat_SparseElemental *cliq;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Calls to the SparseElemental solvers is not currently implemented");
  ierr = PetscElementalInitializePackage();CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscStrallocpy("sparseelememtal",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_CHOLESKY){
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SparseElemental;
    B->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_SparseElemental;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");

  ierr = PetscNewLog(B,&cliq);CHKERRQ(ierr);
  B->data            = (void*)cliq;
  El::mpi::Comm cxxcomm(PetscObjectComm((PetscObject)A));
  ierr = PetscCommDuplicate(PetscObjectComm((PetscObject)A),&cliq->comm,NULL);CHKERRQ(ierr);
  cliq->rhs           = new El::DistMultiVec<PetscElemScalar>(A->rmap->N,1,cliq->comm);
  cliq->inverseMap    = new El::DistMap;
  cliq->CleanUp       = PETSC_FALSE;

  B->ops->getinfo = MatGetInfo_External;
  B->ops->view    = MatView_SparseElemental;
  B->ops->mult    = MatMult_SparseElemental; /* for cliq->cmat */
  B->ops->solve   = MatSolve_SparseElemental;

  B->ops->destroy = MatDestroy_SparseElemental;
  B->factortype   = ftype;
  B->assembled    = PETSC_FALSE;

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERSPARSEELEMENTAL,&B->solvertype);CHKERRQ(ierr);

  /* Set Clique options */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"Clique Options","Mat");CHKERRQ(ierr);
  cliq->cutoff      = 128;  /* maximum size of leaf node */
  cliq->numDistSeps = 1;    /* number of distributed separators to try */
  cliq->numSeqSeps  = 1;    /* number of sequential separators to try */
  PetscOptionsEnd();

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_SparseElemental(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERSPARSEELEMENTAL,MATMPIAIJ,        MAT_FACTOR_LU,MatGetFactor_aij_sparseelemental);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
