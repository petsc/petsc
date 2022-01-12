
/*
   Provides an interface to the CHOLMOD sparse solver available through SuiteSparse version 4.2.1

   When built with PETSC_USE_64BIT_INDICES this will use Suitesparse_long as the
   integer type in UMFPACK, otherwise it will use int. This means
   all integers in this file as simply declared as PetscInt. Also it means
   that one cannot use 64BIT_INDICES on 32bit machines [as Suitesparse_long is 32bit only]

*/

#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/sbaij/seq/cholmod/cholmodimpl.h>

/*
   This is a terrible hack, but it allows the error handler to retain a context.
   Note that this hack really cannot be made both reentrant and concurrent.
*/
static Mat static_F;

static void CholmodErrorHandler(int status,const char *file,int line,const char *message)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (status > CHOLMOD_OK) {
    ierr = PetscInfo4(static_F,"CHOLMOD warning %d at %s:%d: %s\n",status,file,line,message);CHKERRV(ierr);
  } else if (status == CHOLMOD_OK) { /* Documentation says this can happen, but why? */
    ierr = PetscInfo3(static_F,"CHOLMOD OK at %s:%d: %s\n",file,line,message);CHKERRV(ierr);
  } else {
    ierr = PetscErrorPrintf("CHOLMOD error %d at %s:%d: %s\n",status,file,line,message);CHKERRV(ierr);
  }
  PetscFunctionReturnVoid();
}

PetscErrorCode  CholmodStart(Mat F)
{
  PetscErrorCode ierr;
  Mat_CHOLMOD    *chol=(Mat_CHOLMOD*)F->data;
  cholmod_common *c;
  PetscBool      flg;

  PetscFunctionBegin;
  if (chol->common) PetscFunctionReturn(0);
  ierr = PetscMalloc1(1,&chol->common);CHKERRQ(ierr);
  ierr = !cholmod_X_start(chol->common);CHKERRQ(ierr);

  c                = chol->common;
  c->error_handler = CholmodErrorHandler;

#define CHOLMOD_OPTION_DOUBLE(name,help) do {                            \
    PetscReal tmp = (PetscReal)c->name;                                  \
    ierr    = PetscOptionsReal("-mat_cholmod_" #name,help,"None",tmp,&tmp,NULL);CHKERRQ(ierr); \
    c->name = (double)tmp;                                               \
} while (0)

#define CHOLMOD_OPTION_INT(name,help) do {                               \
    PetscInt tmp = (PetscInt)c->name;                                    \
    ierr    = PetscOptionsInt("-mat_cholmod_" #name,help,"None",tmp,&tmp,NULL);CHKERRQ(ierr); \
    c->name = (int)tmp;                                                  \
} while (0)

#define CHOLMOD_OPTION_SIZE_T(name,help) do {                            \
    PetscReal tmp = (PetscInt)c->name;                                   \
    ierr = PetscOptionsReal("-mat_cholmod_" #name,help,"None",tmp,&tmp,NULL);CHKERRQ(ierr); \
    if (tmp < 0) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_OUTOFRANGE,"value must be positive"); \
    c->name = (size_t)tmp;                                               \
} while (0)

#define CHOLMOD_OPTION_BOOL(name,help) do {                             \
    PetscBool tmp = (PetscBool) !!c->name;                              \
    ierr    = PetscOptionsBool("-mat_cholmod_" #name,help,"None",tmp,&tmp,NULL);CHKERRQ(ierr); \
    c->name = (int)tmp;                                                  \
} while (0)

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)F),((PetscObject)F)->prefix,"CHOLMOD Options","Mat");CHKERRQ(ierr);
  CHOLMOD_OPTION_INT(nmethods,"Number of different ordering methods to try");

#if defined(PETSC_USE_SUITESPARSE_GPU)
  c->useGPU = 1;
  CHOLMOD_OPTION_INT(useGPU,"Use GPU for BLAS 1, otherwise 0");
  CHOLMOD_OPTION_SIZE_T(maxGpuMemBytes,"Maximum memory to allocate on the GPU");
  CHOLMOD_OPTION_DOUBLE(maxGpuMemFraction,"Fraction of available GPU memory to allocate");
#endif

  /* CHOLMOD handles first-time packing and refactor-packing separately, but we usually want them to be the same. */
  chol->pack = (PetscBool)c->final_pack;
  ierr = PetscOptionsBool("-mat_cholmod_pack","Pack factors after factorization [disable for frequent repeat factorization]","None",chol->pack,&chol->pack,NULL);CHKERRQ(ierr);
  c->final_pack = (int)chol->pack;

  CHOLMOD_OPTION_DOUBLE(dbound,"Minimum absolute value of diagonal entries of D");
  CHOLMOD_OPTION_DOUBLE(grow0,"Global growth ratio when factors are modified");
  CHOLMOD_OPTION_DOUBLE(grow1,"Column growth ratio when factors are modified");
  CHOLMOD_OPTION_SIZE_T(grow2,"Affine column growth constant when factors are modified");
  CHOLMOD_OPTION_SIZE_T(maxrank,"Max rank of update, larger values are faster but use more memory [2,4,8]");
  {
    static const char *const list[] = {"SIMPLICIAL","AUTO","SUPERNODAL","MatCholmodFactorType","MAT_CHOLMOD_FACTOR_",0};
    ierr = PetscOptionsEnum("-mat_cholmod_factor","Factorization method","None",list,(PetscEnum)c->supernodal,(PetscEnum*)&c->supernodal,NULL);CHKERRQ(ierr);
  }
  if (c->supernodal) CHOLMOD_OPTION_DOUBLE(supernodal_switch,"flop/nnz_L threshold for switching to supernodal factorization");
  CHOLMOD_OPTION_BOOL(final_asis,"Leave factors \"as is\"");
  CHOLMOD_OPTION_BOOL(final_pack,"Pack the columns when finished (use FALSE if the factors will be updated later)");
  if (!c->final_asis) {
    CHOLMOD_OPTION_BOOL(final_super,"Leave supernodal factors instead of converting to simplicial");
    CHOLMOD_OPTION_BOOL(final_ll,"Turn LDL' factorization into LL'");
    CHOLMOD_OPTION_BOOL(final_monotonic,"Ensure columns are monotonic when done");
    CHOLMOD_OPTION_BOOL(final_resymbol,"Remove numerically zero values resulting from relaxed supernodal amalgamation");
  }
  {
    PetscReal tmp[] = {(PetscReal)c->zrelax[0],(PetscReal)c->zrelax[1],(PetscReal)c->zrelax[2]};
    PetscInt  n     = 3;
    ierr = PetscOptionsRealArray("-mat_cholmod_zrelax","3 real supernodal relaxed amalgamation parameters","None",tmp,&n,&flg);CHKERRQ(ierr);
    if (flg && n != 3) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_OUTOFRANGE,"must provide exactly 3 parameters to -mat_cholmod_zrelax");
    if (flg) while (n--) c->zrelax[n] = (double)tmp[n];
  }
  {
    PetscInt n,tmp[] = {(PetscInt)c->nrelax[0],(PetscInt)c->nrelax[1],(PetscInt)c->nrelax[2]};
    ierr = PetscOptionsIntArray("-mat_cholmod_nrelax","3 size_t supernodal relaxed amalgamation parameters","None",tmp,&n,&flg);CHKERRQ(ierr);
    if (flg && n != 3) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_OUTOFRANGE,"must provide exactly 3 parameters to -mat_cholmod_nrelax");
    if (flg) while (n--) c->nrelax[n] = (size_t)tmp[n];
  }
  CHOLMOD_OPTION_BOOL(prefer_upper,"Work with upper triangular form [faster when using fill-reducing ordering, slower in natural ordering]");
  CHOLMOD_OPTION_BOOL(default_nesdis,"Use NESDIS instead of METIS for nested dissection");
  CHOLMOD_OPTION_INT(print,"Verbosity level");
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatWrapCholmod_seqsbaij(Mat A,PetscBool values,cholmod_sparse *C,PetscBool *aijalloc,PetscBool *valloc)
{
  Mat_SeqSBAIJ   *sbaij = (Mat_SeqSBAIJ*)A->data;
  PetscBool      vallocin = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(C,sizeof(*C));CHKERRQ(ierr);
  /* CHOLMOD uses column alignment, SBAIJ stores the upper factor, so we pass it on as a lower factor, swapping the meaning of row and column */
  C->nrow   = (size_t)A->cmap->n;
  C->ncol   = (size_t)A->rmap->n;
  C->nzmax  = (size_t)sbaij->maxnz;
  C->p      = sbaij->i;
  C->i      = sbaij->j;
  if (values) {
#if defined(PETSC_USE_COMPLEX)
    /* we need to pass CHOLMOD the conjugate matrix */
    PetscScalar *v;
    PetscInt    i;

    ierr = PetscMalloc1(sbaij->maxnz,&v);CHKERRQ(ierr);
    for (i = 0; i < sbaij->maxnz; i++) v[i] = PetscConj(sbaij->a[i]);
    C->x = v;
    vallocin = PETSC_TRUE;
#else
    C->x = sbaij->a;
#endif
  }
  C->stype  = -1;
  C->itype  = CHOLMOD_INT_TYPE;
  C->xtype  = values ? CHOLMOD_SCALAR_TYPE : CHOLMOD_PATTERN;
  C->dtype  = CHOLMOD_DOUBLE;
  C->sorted = 1;
  C->packed = 1;
  *aijalloc = PETSC_FALSE;
  *valloc   = vallocin;
  PetscFunctionReturn(0);
}

#define GET_ARRAY_READ 0
#define GET_ARRAY_WRITE 1

PetscErrorCode VecWrapCholmod(Vec X,PetscInt rw,cholmod_dense *Y)
{
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscInt       n;

  PetscFunctionBegin;
  ierr = PetscMemzero(Y,sizeof(*Y));CHKERRQ(ierr);
  switch (rw) {
  case GET_ARRAY_READ:
    ierr = VecGetArrayRead(X,(const PetscScalar**)&x);CHKERRQ(ierr);
    break;
  case GET_ARRAY_WRITE:
    ierr = VecGetArrayWrite(X,&x);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Case %" PetscInt_FMT " not handled",rw);
    break;
  }
  ierr = VecGetSize(X,&n);CHKERRQ(ierr);

  Y->x     = x;
  Y->nrow  = n;
  Y->ncol  = 1;
  Y->nzmax = n;
  Y->d     = n;
  Y->xtype = CHOLMOD_SCALAR_TYPE;
  Y->dtype = CHOLMOD_DOUBLE;
  PetscFunctionReturn(0);
}

PetscErrorCode VecUnWrapCholmod(Vec X,PetscInt rw,cholmod_dense *Y)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  switch (rw) {
  case GET_ARRAY_READ:
    ierr = VecRestoreArrayRead(X,(const PetscScalar**)&Y->x);CHKERRQ(ierr);
    break;
  case GET_ARRAY_WRITE:
    ierr = VecRestoreArrayWrite(X,(PetscScalar**)&Y->x);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Case %" PetscInt_FMT " not handled",rw);
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseWrapCholmod(Mat X,PetscInt rw,cholmod_dense *Y)
{
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscInt       m,n,lda;

  PetscFunctionBegin;
  ierr = PetscMemzero(Y,sizeof(*Y));CHKERRQ(ierr);
  switch (rw) {
  case GET_ARRAY_READ:
    ierr = MatDenseGetArrayRead(X,(const PetscScalar**)&x);CHKERRQ(ierr);
    break;
  case GET_ARRAY_WRITE:
    ierr = MatDenseGetArrayWrite(X,&x);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Case %" PetscInt_FMT " not handled",rw);
    break;
  }
  ierr = MatDenseGetLDA(X,&lda);CHKERRQ(ierr);
  ierr = MatGetLocalSize(X,&m,&n);CHKERRQ(ierr);

  Y->x     = x;
  Y->nrow  = m;
  Y->ncol  = n;
  Y->nzmax = lda*n;
  Y->d     = lda;
  Y->xtype = CHOLMOD_SCALAR_TYPE;
  Y->dtype = CHOLMOD_DOUBLE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseUnWrapCholmod(Mat X,PetscInt rw,cholmod_dense *Y)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  switch (rw) {
  case GET_ARRAY_READ:
    ierr = MatDenseRestoreArrayRead(X,(const PetscScalar**)&Y->x);CHKERRQ(ierr);
    break;
  case GET_ARRAY_WRITE:
    /* we don't have MatDenseRestoreArrayWrite */
    ierr = MatDenseRestoreArray(X,(PetscScalar**)&Y->x);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Case %" PetscInt_FMT " not handled",rw);
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode  MatDestroy_CHOLMOD(Mat F)
{
  PetscErrorCode ierr;
  Mat_CHOLMOD    *chol=(Mat_CHOLMOD*)F->data;

  PetscFunctionBegin;
  if (chol->spqrfact) {
    ierr = !SuiteSparseQR_C_free(&chol->spqrfact, chol->common);CHKERRQ(ierr);
  }
  if (chol->factor) {
    ierr = !cholmod_X_free_factor(&chol->factor,chol->common);CHKERRQ(ierr);
  }
  if (chol->common->itype == CHOLMOD_INT) {
    ierr = !cholmod_finish(chol->common);CHKERRQ(ierr);
  } else {
    ierr = !cholmod_l_finish(chol->common);CHKERRQ(ierr);
  }
  ierr = PetscFree(chol->common);CHKERRQ(ierr);
  ierr = PetscFree(chol->matrix);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)F,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(F->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_CHOLMOD(Mat,Vec,Vec);
static PetscErrorCode MatMatSolve_CHOLMOD(Mat,Mat,Mat);

/*static const char *const CholmodOrderingMethods[] = {"User","AMD","METIS","NESDIS(default)","Natural","NESDIS(small=20000)","NESDIS(small=4,no constrained)","NESDIS()"};*/

static PetscErrorCode MatView_Info_CHOLMOD(Mat F,PetscViewer viewer)
{
  Mat_CHOLMOD          *chol = (Mat_CHOLMOD*)F->data;
  const cholmod_common *c    = chol->common;
  PetscErrorCode       ierr;
  PetscInt             i;

  PetscFunctionBegin;
  if (F->ops->solve != MatSolve_CHOLMOD) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPrintf(viewer,"CHOLMOD run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Pack factors after symbolic factorization: %s\n",chol->pack ? "TRUE" : "FALSE");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.dbound            %g  (Smallest absolute value of diagonal entries of D)\n",c->dbound);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.grow0             %g\n",c->grow0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.grow1             %g\n",c->grow1);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.grow2             %u\n",(unsigned)c->grow2);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.maxrank           %u\n",(unsigned)c->maxrank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.supernodal_switch %g\n",c->supernodal_switch);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.supernodal        %d\n",c->supernodal);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_asis        %d\n",c->final_asis);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_super       %d\n",c->final_super);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_ll          %d\n",c->final_ll);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_pack        %d\n",c->final_pack);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_monotonic   %d\n",c->final_monotonic);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_resymbol    %d\n",c->final_resymbol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.zrelax            [%g,%g,%g]\n",c->zrelax[0],c->zrelax[1],c->zrelax[2]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.nrelax            [%u,%u,%u]\n",(unsigned)c->nrelax[0],(unsigned)c->nrelax[1],(unsigned)c->nrelax[2]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.prefer_upper      %d\n",c->prefer_upper);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.print             %d\n",c->print);CHKERRQ(ierr);
  for (i=0; i<c->nmethods; i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"Ordering method %" PetscInt_FMT "%s:\n",i,i==c->selected ? " [SELECTED]" : "");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  lnz %g, fl %g, prune_dense %g, prune_dense2 %g\n",
                                  c->method[i].lnz,c->method[i].fl,c->method[i].prune_dense,c->method[i].prune_dense2);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"Common.postorder         %d\n",c->postorder);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.default_nesdis    %d (use NESDIS instead of METIS for nested dissection)\n",c->default_nesdis);CHKERRQ(ierr);
  /* Statistics */
  ierr = PetscViewerASCIIPrintf(viewer,"Common.fl                %g (flop count from most recent analysis)\n",c->fl);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.lnz               %g (fundamental nz in L)\n",c->lnz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.anz               %g\n",c->anz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.modfl             %g (flop count from most recent update)\n",c->modfl);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.malloc_count      %g (number of live objects)\n",(double)c->malloc_count);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.memory_usage      %g (peak memory usage in bytes)\n",(double)c->memory_usage);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.memory_inuse      %g (current memory usage in bytes)\n",(double)c->memory_inuse);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.nrealloc_col      %g (number of column reallocations)\n",c->nrealloc_col);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.nrealloc_factor   %g (number of factor reallocations due to column reallocations)\n",c->nrealloc_factor);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.ndbounds_hit      %g (number of times diagonal was modified by dbound)\n",c->ndbounds_hit);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.rowfacfl          %g (number of flops in last call to cholmod_rowfac)\n",c->rowfacfl);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.aatfl             %g (number of flops to compute A(:,f)*A(:,f)')\n",c->aatfl);CHKERRQ(ierr);
#if defined(PETSC_USE_SUITESPARSE_GPU)
  ierr = PetscViewerASCIIPrintf(viewer,"Common.useGPU            %d\n",c->useGPU);CHKERRQ(ierr);
#endif
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode  MatView_CHOLMOD(Mat F,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatView_Info_CHOLMOD(F,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_CHOLMOD(Mat F,Vec B,Vec X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,cholX,*X_handle,*Y_handle = NULL,*E_handle = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  static_F = F;
  ierr = VecWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  ierr = VecWrapCholmod(X,GET_ARRAY_WRITE,&cholX);CHKERRQ(ierr);
  X_handle = &cholX;
  ierr = !cholmod_X_solve2(CHOLMOD_A,chol->factor,&cholB,NULL,&X_handle,NULL,&Y_handle,&E_handle,chol->common);CHKERRQ(ierr);
  ierr = !cholmod_X_free_dense(&Y_handle,chol->common);CHKERRQ(ierr);
  ierr = !cholmod_X_free_dense(&E_handle,chol->common);CHKERRQ(ierr);
  ierr = VecUnWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  ierr = VecUnWrapCholmod(X,GET_ARRAY_WRITE,&cholX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_CHOLMOD(Mat F,Mat B,Mat X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,cholX,*X_handle,*Y_handle = NULL,*E_handle = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  static_F = F;
  ierr = MatDenseWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  ierr = MatDenseWrapCholmod(X,GET_ARRAY_WRITE,&cholX);CHKERRQ(ierr);
  X_handle = &cholX;
  ierr = !cholmod_X_solve2(CHOLMOD_A,chol->factor,&cholB,NULL,&X_handle,NULL,&Y_handle,&E_handle,chol->common);CHKERRQ(ierr);
  ierr = !cholmod_X_free_dense(&Y_handle,chol->common);CHKERRQ(ierr);
  ierr = !cholmod_X_free_dense(&E_handle,chol->common);CHKERRQ(ierr);
  ierr = MatDenseUnWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  ierr = MatDenseUnWrapCholmod(X,GET_ARRAY_WRITE,&cholX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_CHOLMOD(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_sparse cholA;
  PetscBool      aijalloc,valloc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr     = (*chol->Wrap)(A,PETSC_TRUE,&cholA,&aijalloc,&valloc);CHKERRQ(ierr);
  static_F = F;
  ierr     = !cholmod_X_factorize(&cholA,chol->factor,chol->common);
  if (ierr) SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_LIB,"CHOLMOD factorization failed with status %d",chol->common->status);
  if (chol->common->status == CHOLMOD_NOT_POSDEF) SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_MAT_CH_ZRPVT,"CHOLMOD detected that the matrix is not positive definite, failure at column %u",(unsigned)chol->factor->minor);

  if (aijalloc) {ierr = PetscFree2(cholA.p,cholA.i);CHKERRQ(ierr);}
  if (valloc) {ierr = PetscFree(cholA.x);CHKERRQ(ierr);}
#if defined(PETSC_USE_SUITESPARSE_GPU)
  ierr = PetscLogGpuTimeAdd(chol->common->CHOLMOD_GPU_GEMM_TIME + chol->common->CHOLMOD_GPU_SYRK_TIME + chol->common->CHOLMOD_GPU_TRSM_TIME + chol->common->CHOLMOD_GPU_POTRF_TIME);CHKERRQ(ierr);
#endif

  F->ops->solve             = MatSolve_CHOLMOD;
  F->ops->solvetranspose    = MatSolve_CHOLMOD;
  F->ops->matsolve          = MatMatSolve_CHOLMOD;
  F->ops->matsolvetranspose = MatMatSolve_CHOLMOD;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode  MatCholeskyFactorSymbolic_CHOLMOD(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  PetscErrorCode ierr;
  cholmod_sparse cholA;
  PetscBool      aijalloc,valloc;
  PetscInt       *fset = 0;
  size_t         fsize = 0;

  PetscFunctionBegin;
  ierr     = (*chol->Wrap)(A,PETSC_FALSE,&cholA,&aijalloc,&valloc);CHKERRQ(ierr);
  static_F = F;
  if (chol->factor) {
    ierr = !cholmod_X_resymbol(&cholA,fset,fsize,(int)chol->pack,chol->factor,chol->common);
    if (ierr) SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_LIB,"CHOLMOD analysis failed with status %d",chol->common->status);
  } else if (perm) {
    const PetscInt *ip;
    ierr         = ISGetIndices(perm,&ip);CHKERRQ(ierr);
    chol->factor = cholmod_X_analyze_p(&cholA,(PetscInt*)ip,fset,fsize,chol->common);
    if (!chol->factor) SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_LIB,"CHOLMOD analysis failed using PETSc ordering with status %d",chol->common->status);
    ierr = ISRestoreIndices(perm,&ip);CHKERRQ(ierr);
  } else {
    chol->factor = cholmod_X_analyze(&cholA,chol->common);
    if (!chol->factor) SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_LIB,"CHOLMOD analysis failed using internal ordering with status %d",chol->common->status);
  }

  if (aijalloc) {ierr = PetscFree2(cholA.p,cholA.i);CHKERRQ(ierr);}
  if (valloc) {ierr = PetscFree(cholA.x);CHKERRQ(ierr);}

  F->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_CHOLMOD;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_seqsbaij_cholmod(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCHOLMOD;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetInfo_CHOLMOD(Mat F,MatInfoType flag,MatInfo *info)
{
  Mat_CHOLMOD *chol = (Mat_CHOLMOD*)F->data;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = chol->common->lnz;
  info->nz_used           = chol->common->lnz;
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = chol->common->memory_inuse;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = chol->common->malloc_count;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERCHOLMOD

  A matrix type providing direct solvers (Cholesky) for sequential matrices
  via the external package CHOLMOD.

  Use ./configure --download-suitesparse to install PETSc to use CHOLMOD

  Use -pc_type cholesky -pc_factor_mat_solver_type cholmod to use this direct solver

  Consult CHOLMOD documentation for more information about the Common parameters
  which correspond to the options database keys below.

  Options Database Keys:
+ -mat_cholmod_dbound <0>          - Minimum absolute value of diagonal entries of D (None)
. -mat_cholmod_grow0 <1.2>         - Global growth ratio when factors are modified (None)
. -mat_cholmod_grow1 <1.2>         - Column growth ratio when factors are modified (None)
. -mat_cholmod_grow2 <5>           - Affine column growth constant when factors are modified (None)
. -mat_cholmod_maxrank <8>         - Max rank of update, larger values are faster but use more memory [2,4,8] (None)
. -mat_cholmod_factor <AUTO>       - (choose one of) SIMPLICIAL AUTO SUPERNODAL
. -mat_cholmod_supernodal_switch <40> - flop/nnz_L threshold for switching to supernodal factorization (None)
. -mat_cholmod_final_asis <TRUE>   - Leave factors "as is" (None)
. -mat_cholmod_final_pack <TRUE>   - Pack the columns when finished (use FALSE if the factors will be updated later) (None)
. -mat_cholmod_zrelax <0.8>        - 3 real supernodal relaxed amalgamation parameters (None)
. -mat_cholmod_nrelax <4>          - 3 size_t supernodal relaxed amalgamation parameters (None)
. -mat_cholmod_prefer_upper <TRUE> - Work with upper triangular form (faster when using fill-reducing ordering, slower in natural ordering) (None)
. -mat_cholmod_print <3>           - Verbosity level (None)
- -mat_ordering_type internal      - Use the ordering provided by Cholmod

   Level: beginner

   Note: CHOLMOD is part of SuiteSparse http://faculty.cse.tamu.edu/davis/suitesparse.html

.seealso: PCCHOLESKY, PCFactorSetMatSolverType(), MatSolverType
M*/

PETSC_INTERN PetscErrorCode MatGetFactor_seqsbaij_cholmod(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_CHOLMOD    *chol;
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=A->cmap->n,bs;
  const char     *prefix;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (bs != 1) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"CHOLMOD only supports block size=1, given %" PetscInt_FMT,bs);
#if defined(PETSC_USE_COMPLEX)
  if (!A->hermitian) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only for Hermitian matrices");
#endif
  /* Create the factorization matrix F */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = PetscStrallocpy("cholmod",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(B,prefix);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = PetscNewLog(B,&chol);CHKERRQ(ierr);

  chol->Wrap    = MatWrapCholmod_seqsbaij;
  B->data       = chol;

  B->ops->getinfo                = MatGetInfo_CHOLMOD;
  B->ops->view                   = MatView_CHOLMOD;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_CHOLMOD;
  B->ops->destroy                = MatDestroy_CHOLMOD;
  ierr                           = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_seqsbaij_cholmod);CHKERRQ(ierr);
  B->factortype                  = MAT_FACTOR_CHOLESKY;
  B->assembled                   = PETSC_TRUE;
  B->preallocated                = PETSC_TRUE;

  ierr = CholmodStart(B);CHKERRQ(ierr);

  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERCHOLMOD,&B->solvertype);CHKERRQ(ierr);
  B->canuseordering = PETSC_TRUE;
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_CHOLESKY]);CHKERRQ(ierr);
  *F   = B;
  PetscFunctionReturn(0);
}
