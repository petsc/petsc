#define PETSCKSP_DLL

#include "src/ksp/pc/impls/schur/schur.h"      /*I "petscpc.h" I*/
#include "src/ksp/pc/impls/schur/sppack.h"
#include "private/matimpl.h"

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
#define VecScatterBegin(sct,x,y,im,sm) \
        VecScatterBegin(x,y,im,sm,sct)
#define VecScatterEnd(sct,x,y,im,sm) \
        VecScatterEnd(x,y,im,sm,sct)
#define MatGetRowIJ(madj,shift,symm,blkcomp,n,ia,ja,done) \
        MatGetRowIJ(madj,shift,symm,n,ia,ja,done)
#define MatRestoreRowIJ(madj,shift,symm,blkcomp,n,ia,ja,done) \
        MatRestoreRowIJ(madj,shift,symm,n,ia,ja,done)
#endif


#if !defined(PetscMallocInt)
#define PetscMallocInt(n,p) PetscMalloc((n)*sizeof(PetscInt),(p))
#endif

/* -------------------------------------------------------------------------- */

/* Schur Complement Operator */
/*  ------------------------ */

#undef __FUNCT__  
#define __FUNCT__ "PCSchur_ApplySchur_Seq"
static PetscErrorCode PCSchur_ApplySchur_Seq(PC_Schur *ctx, 
					     Vec x_B, Vec y_B,
					     PetscTruth transposed)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!transposed) { /* apply local Schur complement */
    /* y_B  <- (A_BB - A_BI * A_II^{-1} * A_IB) * x_B */
    ierr = MatMult(ctx->A_BB,x_B,y_B);CHKERRQ(ierr);
    ierr = MatMult(ctx->A_IB,x_B,ctx->vec1_I);CHKERRQ(ierr);
    ierr = KSPSolve(ctx->ksp_I,ctx->vec1_I,ctx->vec2_I);CHKERRQ(ierr);
    ierr = MatMult(ctx->A_BI,ctx->vec2_I,ctx->vec1_B);CHKERRQ(ierr);
    ierr = VecAXPY(y_B,-1,ctx->vec1_B);CHKERRQ(ierr);
  } else {           /* apply transposed local Schur complement */
    /* y_B  <- (A_BB' - A_IB' * A_II^{-1}' * A_BI') * x_B */
    ierr = MatMultTranspose(ctx->A_BB,x_B,y_B);CHKERRQ(ierr);
    ierr = MatMultTranspose(ctx->A_BI,x_B,ctx->vec1_I);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(ctx->ksp_I,ctx->vec1_I,ctx->vec2_I);CHKERRQ(ierr);
    ierr = MatMultTranspose(ctx->A_IB,ctx->vec2_I,ctx->vec1_B);CHKERRQ(ierr);
    ierr = VecAXPY(y_B,-1,ctx->vec1_B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSchur_AddOffDiag_MPI"
static PetscErrorCode PCSchur_AddOffDiag_MPI(PC_Schur *ctx, 
					     Vec x_B, Vec y_B,
					     PetscTruth transposed)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = VecSet(ctx->vec_G,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->G_to_B,x_B,ctx->vec_G,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (ctx->G_to_B,x_B,ctx->vec_G,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  if (!transposed) { 
    ierr = VecSet(ctx->lvec,0);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->Mvctx,ctx->vec_G,ctx->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (ctx->Mvctx,ctx->vec_G,ctx->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecSet(ctx->vec_G,0);CHKERRQ(ierr); /* XXX Why is this needed ??? */
    ierr = (*ctx->B->ops->mult)(ctx->B,ctx->lvec,ctx->vec_G);CHKERRQ(ierr);
  } else {
    ierr = VecSet(ctx->lvec,0);CHKERRQ(ierr);
    ierr = (*ctx->B->ops->multtranspose)(ctx->B,ctx->vec_G,ctx->lvec);CHKERRQ(ierr);
    ierr = VecSet(ctx->vec_G,0);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->Mvctx,ctx->lvec,ctx->vec_G,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (ctx->Mvctx,ctx->lvec,ctx->vec_G,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }

  ierr = VecScatterBegin(ctx->G_to_B,ctx->vec_G,y_B,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (ctx->G_to_B,ctx->vec_G,y_B,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSchur_ApplySchur_MPI"
static PetscErrorCode PCSchur_ApplySchur_MPI(PC_Schur *ctx,
					     Vec x_S,Vec y_S,
					     PetscTruth transposed)
{
  Vec            x_B,y_B;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  x_B = ctx->vec2_B;
  y_B = ctx->vec3_B;
  /* gather from global interface to local interface */
  /* x_B  <- x[B] */
  ierr = VecScatterBegin(ctx->S_to_B,x_S,x_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (ctx->S_to_B,x_S,x_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* apply local Schur complement */
  ierr = PCSchur_ApplySchur_Seq(ctx,x_B,y_B,transposed);CHKERRQ(ierr);
  /* add off-processor contributions */
  ierr = PCSchur_AddOffDiag_MPI(ctx,x_B,y_B,transposed);CHKERRQ(ierr);
  /* scatter from local interface to global interface */
  /* y[B] <- y_B */
  ierr = VecSet(y_S,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->S_to_B,y_B,y_S,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (ctx->S_to_B,y_B,y_S,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSchur_ApplySchur"
static PetscErrorCode PCSchur_ApplySchur(PC_Schur *ctx,
					 Vec x,Vec y,
					 PetscTruth transposed)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (ctx->seq) { ierr = PCSchur_ApplySchur_Seq(ctx,x,y,transposed);CHKERRQ(ierr); }
  else          { ierr = PCSchur_ApplySchur_MPI(ctx,x,y,transposed);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Schur"
static PetscErrorCode MatMult_Schur(Mat S,Vec x,Vec y)
{
  PC_Schur       *ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&ctx);CHKERRQ(ierr);
  ierr = PCSchur_ApplySchur(ctx,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Schur"
static PetscErrorCode MatMultTranspose_Schur(Mat S,Vec x,Vec y)
{
  PC_Schur       *ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&ctx);CHKERRQ(ierr);
  ierr = PCSchur_ApplySchur(ctx,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_Schur"
static PetscErrorCode MatGetDiagonal_Schur(Mat S,Vec v)
{
  PC_Schur       *ctx;
  Vec            diag;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&ctx);CHKERRQ(ierr);
  /* get diagonal of local interface matrix */
  diag = ctx->seq ? v : ctx->vec1_B;
  ierr = MatGetDiagonal(ctx->A_BB,diag);CHKERRQ(ierr);
  if (ctx->seq) PetscFunctionReturn(0);
  /* scatter from local interface to global interface */
  ierr = VecSet(v,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->S_to_B,diag,v,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (ctx->S_to_B,diag,v,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonalBlock_Schur"
static PetscErrorCode MatGetDiagonalBlock_Schur(Mat S,PetscTruth *iscopy,MatReuse reuse,Mat *Sd)
{
  PC_Schur       *ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,MAT_COOKIE,1);
  PetscValidPointer(iscopy,2);
  PetscValidPointer(Sd,4);
  ierr = MatShellGetContext(S,(void**)&ctx);CHKERRQ(ierr);
  *Sd      = ctx->A_BB;
  *iscopy = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_Schur"
static PetscErrorCode MatView_Schur(Mat S,PetscViewer viewer)
{
  PC_Schur       *ctx;
  PetscTruth     iascii;
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (!iascii) {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)viewer)->type_name);
    PetscFunctionReturn(PETSC_ERR_SUP);
  }
  ierr = MatShellGetContext(S,(void**)&ctx);CHKERRQ(ierr);
  if(!ctx->sp_stats) PetscFunctionReturn(0);
  ierr = MPI_Comm_size(S->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(S->comm,&rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Number of local nodes in each processor for\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"interior (I), interface (B), subdomain (N)\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"-------------------------------------------\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%3d] I = %D, B = %D, N = %D\n",
					    rank,ctx->n_I,ctx->n_B,ctx->n);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* -------------------------------------------------------------------------- */

#undef RGB_FNSEP_ND
#undef RGB_FNSEP_1WD
#undef RGB_FNSEP_HALF

#define RGB_FNSEP_ND   0
#define RGB_FNSEP_1WD  1
#define RGB_FNSEP_HALF 2

#if defined(RGB_FNSEP_METHOD)
#if ((RGB_FNSEP_METHOD < RGB_FNSEP_ND) || (RGB_FNSEP_METHOD > RGB_FNSEP_HALF))
#undef RGB_FNSEP_METHOD
#endif
#endif

#if !defined(RGB_FNSEP_METHOD)
#define RGB_FNSEP_METHOD  RGB_FNSEP_ND
#endif

#undef __FUNCT__  
#define __FUNCT__ "PCSchur_Separator"
static PetscErrorCode PCSchur_Separator_RGB(PC_Schur *schur, PetscInt maxsize, 
					    PetscInt *nidx, PetscInt indices[])
{
  Mat            madj=PETSC_NULL;
  
  PetscInt       i,j;
  PetscTruth     done;
  
  PetscInt       *work;
  PetscInt       neqns,*xadj,*adjncy;
  PetscInt       root, *mask;
  PetscInt       lvl,nlvl,*xls,*ls;
  PetscInt       n,nsep,*sep;
  PetscInt       ccsize;

  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(schur,1);

  /* get symmetric matrix graph with indices starting at 1  */
#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
  ierr = MatConvert(schur->A,MATMPIADJ,MAT_INITIAL_MATRIX,&madj);CHKERRQ(ierr);
#else
  ierr = PetscObjectReference((PetscObject)schur->A);CHKERRQ(ierr);
  madj = schur->A;
#endif
  ierr = MatGetRowIJ(madj,1,PETSC_TRUE,PETSC_FALSE,&neqns,&xadj,&adjncy,&done);CHKERRQ(ierr);
  if (!done) {
    ierr = MatDestroy(madj);CHKERRQ(ierr);
    SETERRQ1(PETSC_ERR_SUP,"Cannot get IJ structure for matrix type %s",((PetscObject)madj)->type_name);
    PetscFunctionReturn(PETSC_ERR_SUP);
  }
  /* workspace allocation */
  ierr = PetscMalloc((4*neqns+1)*sizeof(PetscInt),&work);CHKERRQ(ierr);
  mask = work;
  xls  = mask + neqns;
  ls   = xls  + neqns + 1;
  sep  = ls   + neqns;
  /* adjustments */
  --mask; --xls; --ls; --sep;

  schur->sp_count = 0;
  schur->sp_minsz = neqns;
  schur->sp_maxsz = 1;
  schur->sp_nseps = 0;

  /* recursive graph bisection */
  n = 0; for (i=1; i<=neqns; i++) mask[i]=1;
  for (i=1; i<=neqns; i++) {
    if (mask[i] == 0) continue;
    root = i;
    ierr = SPARSEPACKrootls(&root,xadj,adjncy,&mask[1],&nlvl,&xls[1],&ls[1]);CHKERRQ(ierr);
    ccsize = xls[nlvl+1] - 1;
    if (nlvl > 2 && ccsize > maxsize) {
      PetscInt method = RGB_FNSEP_METHOD;
      switch (method) {
      case RGB_FNSEP_ND:
	ierr = SPARSEPACKfndsep(&root,xadj,adjncy,&mask[1],&nsep,&sep[1+n],&xls[1],&ls[1]);CHKERRQ(ierr); 
	break;
      case RGB_FNSEP_1WD:
	ierr = SPARSEPACKfn1wd(&root,xadj,adjncy,&mask[1],&nsep,&sep[1+n],&nlvl,&xls[1],&ls[1]);CHKERRQ(ierr);
	break;
      case RGB_FNSEP_HALF:
	lvl = 1; nsep = 0; 
	while ((ccsize/2) > (xls[lvl]-1)) lvl++;
	for (j=xls[lvl]; j<=xls[lvl+1]; j++) { 
	  mask[ls[j]] = 0; sep[1+n+nsep] = ls[j]; ++nsep; 
	}
	break;
      default: /* should not fall here */
	nsep = 0;
	break;
      }
      if ((ccsize-nsep) > maxsize) 
	{ schur->sp_nseps += nsep; n += nsep; continue; }
    }
    /* we reached here because this connected component is small enough, 
       so mask their nodes because it does not need further partitioning  */
    for (j=1; j<=ccsize; j++) mask[ls[j]] = 0;
    schur->sp_count = schur->sp_count + 1;
    schur->sp_minsz = PetscMin(schur->sp_minsz, ccsize);
    schur->sp_maxsz = PetscMax(schur->sp_maxsz, ccsize);
  }
  /* shift because SPARSEPACK indices start at one */
  for (j=1; j<=n; j++) sep[j]-- ;
  /* copy result in output array */
  ierr = PetscMemcpy(indices,&sep[1],n*sizeof(PetscInt));CHKERRQ(ierr);
  *nidx = n;

  ierr = MatRestoreRowIJ(madj,1,PETSC_TRUE,PETSC_FALSE,&neqns,&xadj,&adjncy,&done);CHKERRQ(ierr);
  ierr = MatDestroy(madj);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef RGB_FNSEP_ND
#undef RGB_FNSEP_1WD
#undef RGB_FNSEP_HALF
#undef RGB_FNSEP_METHOD


#undef __FUNCT__  
#define __FUNCT__ "PCSchur_LocalSubpart"
static PetscErrorCode PCSchur_LocalSubpart(PC_Schur *schur, IS *separator)
{
  
  PetscInt       ccsize;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(schur,1);
  
  /* determine maximum number of nodes in a subdomain sub-partition */
  if (schur->blocks != PETSC_DECIDE) {
    PetscInt blocks = PetscMin(schur->blocks, schur->n);
    ccsize = schur->n / blocks + schur->n % blocks;
    if (schur->ccsize != PETSC_DECIDE) ccsize = PetscMin(ccsize, schur->ccsize);
  } else if (schur->ccsize == PETSC_DECIDE) {
    /* - in the sequential case, defaults to sub-partitioning in two halves */
    /* - in the parallel case, defaults to no sub-partitioning */
    if (schur->seq) 
      ccsize = schur->n / 2 + schur->n % 2;
    else            
      ccsize = schur->n;
  } else {
    ccsize = schur->ccsize;
  }
  if (ccsize < schur->n) {
    /* determine a graph separator */
    PetscInt n_sep=0,*idx_sep=PETSC_NULL;
    ierr = PetscMallocInt(schur->n,&idx_sep);CHKERRQ(ierr);
    ierr = PCSchur_Separator_RGB(schur,ccsize,&n_sep,idx_sep);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_sep,idx_sep,separator);CHKERRQ(ierr);
    ierr = PetscFree(idx_sep);CHKERRQ(ierr);
    ierr = ISSort(*separator);CHKERRQ(ierr);
  } else {
    /* no subdomain sub-partitioning */
    schur->sp_count = 1;
    schur->sp_minsz = schur->n;
    schur->sp_maxsz = schur->n;
    schur->sp_nseps = 0;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,0,PETSC_NULL,separator);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSchur_GetLocalMatrices"
static PetscErrorCode PCSchur_GetLocalMatrices(PC pc,Mat P,Mat* A,Mat* B,
					       IS* interface,Vec *lvec,VecScatter* Mvctx)
{
  PC_Schur        *schur=(PC_Schur*)(pc->data);
  PetscTruth      is_seqaij,is_mpiaij;
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  
  PetscValidPointer(A,3);
  PetscValidPointer(B,4);
  PetscValidPointer(lvec,6);
  PetscValidPointer(Mvctx,7);
  
  ierr = PetscTypeCompare((PetscObject)P,MATSEQAIJ,&is_seqaij);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)P,MATMPIAIJ,&is_mpiaij);CHKERRQ(ierr);

  if (!(is_seqaij || is_mpiaij)) {
    SETERRQ(PETSC_ERR_ARG_WRONG,
	    "Schur preconditioner requires a matrix of type MAT{SEQ|MPI}AIJ");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONG);
  }

  if (schur->seq) {
    *A = P;
    *B = PETSC_NULL;
    if (interface) {
      ierr = ISCreateGeneral(PETSC_COMM_SELF,0,PETSC_NULL,interface);CHKERRQ(ierr);
    }
    *lvec  = PETSC_NULL;
    *Mvctx = PETSC_NULL;
  } else {
    Mat        Pd,Po;
    PetscInt   *garray;
#if defined (PETSC_USE_CTABLE)
    PetscTable colmap;
#else
    PetscInt   *colmap;
#endif
    ierr = MatMPIAIJGetSeqAIJ(P,&Pd,&Po,&garray);CHKERRQ(ierr);
    *A = Pd;
    *B = Po;
    if (interface) {
      PetscTruth done;
      PetscInt i,n,*ia,*ja;
      PetscInt ng,*ghosts;
      ierr = MatGetRowIJ(Po,0,PETSC_FALSE,PETSC_FALSE,&n,&ia,&ja,&done);CHKERRQ(ierr);
      ierr = PetscMallocInt(n,&ghosts);CHKERRQ(ierr);
      for (ng=0, i=0; i<n; i++) if (ia[i] != ia[i+1]) ghosts[ng++] = i;
      ierr = MatRestoreRowIJ(Po,0,PETSC_FALSE,PETSC_FALSE,&n,&ia,&ja,&done);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ng,ghosts,interface);CHKERRQ(ierr);
      ierr = PetscFree(ghosts);CHKERRQ(ierr);
    }
    ierr = MatGetCommunicationStructs(P,lvec,&colmap,Mvctx);
  }

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
  PCSetFromOptions_Schur - 

   Input Parameter:
.  pc - the preconditioner context

  Application Interface Routine: PCSetFromOptions()
*/

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Schur"
static PetscErrorCode PCSetFromOptions_Schur(PC pc)
{
  PC_Schur       *schur = (PC_Schur*)pc->data;
  PetscInt       blocks;
  PetscInt       ccsize;
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Schur options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_schur_local_blocks","Number of sub-partitions in subdomain",
			 ""/*"PCSchurSetLocalBlocks"*/,schur->blocks,&blocks,&flg);CHKERRQ(ierr);
  if (flg) {
    /*ierr = PCSchurSetLocalBlocks(pc,blocks);CHKERRQ(ierr);*/
    schur->blocks = (blocks>0)?blocks:PETSC_DECIDE;
  }
  ierr = PetscOptionsInt("-pc_schur_local_ccsize","Number of nodes per sub-partition in subdomain",
			 ""/*"PCSchurSetLocalCCSize"*/,schur->ccsize,&ccsize,&flg);CHKERRQ(ierr);
  if (flg) {
    /*ierr = PCSchurSetLocalCCSize(pc,ccsize);CHKERRQ(ierr);*/
    schur->ccsize = (ccsize>0)?ccsize:PETSC_DECIDE;
  }
  ierr = PetscOptionsTruth("-pc_schur_print_stats","Print partitioning statistics in PCView()",
			   "None",schur->sp_stats,&schur->sp_stats,&flg);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/* forward declarations */

EXTERN_C_BEGIN
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Schur(PC);
EXTERN_C_END

static PetscErrorCode PCDestroy_Schur(PC);

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Schur"
/*
   PCSetUp_Schur - Prepares for the use of the Schur preconditioner
                   by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
static PetscErrorCode PCSetUp_Schur(PC pc)
{
  PC_Schur               *schur=(PC_Schur*)(pc->data);
  PetscErrorCode         ierr;
  PetscFunctionBegin;

  
  if (!pc->pmat) {
    SETERRQ(PETSC_ERR_ORDER,"You must call KSPSetOperators() or PCSetOperators() before this call");
    PetscFunctionReturn(PETSC_ERR_ORDER);
  }
  
  if (!pc->setupcalled) {

    PetscMPIInt size,rank;
    PC          sub_pc;
    const char  *prefix;

    IS interface,
       separator,
       subdomain;

    ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    
    if (size > 1) schur->seq = PETSC_FALSE;
    else          schur->seq = PETSC_TRUE;

    /* update local diag and offdiag matrices and other data */
    ierr = PCSchur_GetLocalMatrices(pc,pc->pmat,&schur->A,&schur->B,
				    &interface,&schur->lvec,&schur->Mvctx);CHKERRQ(ierr);
    ierr = MatGetSize(schur->A,&schur->n,PETSC_NULL);CHKERRQ(ierr);
    ierr = ISSort(interface);CHKERRQ(ierr);

    /* subdomain sub-partitioning */
    if (schur->seq) {
      ierr = PCSchur_LocalSubpart(schur, &separator);CHKERRQ(ierr);
    } else {
      IS septmp;
      ierr = PCSchur_LocalSubpart(schur, &septmp);CHKERRQ(ierr);
      ierr = ISDifference(septmp,interface,&separator);CHKERRQ(ierr);
      ierr = ISDestroy(septmp);CHKERRQ(ierr);
    }
    ierr = ISSort(separator);CHKERRQ(ierr);

    /* local index set for interface (B) nodes */
    ierr = ISExpand(interface,separator,&schur->is_B);CHKERRQ(ierr);
    ierr = ISSort(schur->is_B);CHKERRQ(ierr);
    ierr = ISGetSize(schur->is_B,&schur->n_B);CHKERRQ(ierr);
    
    /* local index set for interior (I) nodes */
    ierr = ISCreateStride(PETSC_COMM_SELF,schur->n,0,1,&subdomain);CHKERRQ(ierr);
    ierr = ISDifference(subdomain,schur->is_B,&schur->is_I);CHKERRQ(ierr);
    ierr = ISSort(schur->is_I);CHKERRQ(ierr);
    ierr = ISGetSize(schur->is_I,&schur->n_I);CHKERRQ(ierr);

#if 0
    PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);
    PetscPrintf(PETSC_COMM_SELF,"proc: %D, schur->is_B\n", rank);
    ISView(schur->is_B,0);
    PetscSynchronizedFlush(PETSC_COMM_SELF);
    PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);
#endif

    /* local work vectors for interface (B) nodes */
    ierr = VecCreateSeq(PETSC_COMM_SELF,schur->n_B,&schur->vec1_B);CHKERRQ(ierr);
    ierr = VecDuplicate(schur->vec1_B,&schur->vec2_B);CHKERRQ(ierr);
    ierr = VecDuplicate(schur->vec1_B,&schur->vec3_B);CHKERRQ(ierr);
    
    /* local work vectors for interior (I) nodes */
    ierr = VecCreateSeq(PETSC_COMM_SELF,schur->n_I,&schur->vec1_I);CHKERRQ(ierr);
    ierr = VecDuplicate(schur->vec1_I,&schur->vec2_I);CHKERRQ(ierr);
    ierr = VecDuplicate(schur->vec1_I,&schur->vec3_I);CHKERRQ(ierr);

    /* determine sizes of global interface problem */
    if (schur->seq) {
      schur->n_S = schur->n_B;
      schur->N_S = schur->n_B;
    } else {
      schur->n_S = schur->n_B;
      schur->N_S = PETSC_DECIDE;
      ierr = PetscSplitOwnership(pc->comm,&schur->n_S,&schur->N_S);CHKERRQ(ierr);
    }

    /* work vectors for global interface (S) nodes */
    ierr = VecCreate(pc->comm,&schur->vec1_S);CHKERRQ(ierr);
    ierr = VecSetSizes(schur->vec1_S,schur->n_S,schur->N_S);CHKERRQ(ierr);
    if (schur->seq) { ierr = VecSetType(schur->vec1_S,VECSEQ);CHKERRQ(ierr); }
    else            { ierr = VecSetType(schur->vec1_S,VECMPI);CHKERRQ(ierr); }
    ierr = VecDuplicate(schur->vec1_S,&schur->vec2_S);CHKERRQ(ierr);

    /* vector scatters from all global (G) to local interface (B) and local interior (I) nodes */
    if (schur->seq) {
      Vec vec;
      ierr = MatGetVecs(pc->pmat,PETSC_NULL,&vec);CHKERRQ(ierr);
      ierr = VecScatterCreate(vec,schur->is_B,schur->vec1_B,PETSC_NULL,&schur->G_to_B);CHKERRQ(ierr);
      ierr = VecScatterCreate(vec,schur->is_I,schur->vec1_I,PETSC_NULL,&schur->G_to_I);CHKERRQ(ierr);
      ierr = VecDestroy(vec);CHKERRQ(ierr);
    } else {
      PetscInt               rstart,rend;
      PetscInt               *idx,*idx_G;
      IS                     is_G;
      ISLocalToGlobalMapping map_G;
      /* auxiliar data */
      ierr = MatGetVecs(pc->pmat,PETSC_NULL,&schur->vec_G);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(pc->pmat,&rstart,&rend);CHKERRQ(ierr);
      ierr = ISCreateStride(pc->comm,schur->n,rstart,1,&is_G);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreateIS(is_G,&map_G);CHKERRQ(ierr);
      ierr = PetscMallocInt(PetscMax(schur->n_B,schur->n_I),&idx_G);CHKERRQ(ierr);
      ierr = ISDestroy(is_G);CHKERRQ(ierr);
      /* all global to local interface */
      ierr = ISGetIndices(schur->is_B,&idx);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(map_G,schur->n_B,idx,idx_G);CHKERRQ(ierr);
      ierr = ISRestoreIndices(schur->is_B,&idx);CHKERRQ(ierr);
      ierr = ISCreateGeneralWithArray(PETSC_COMM_SELF,schur->n_B,idx_G,&is_G);CHKERRQ(ierr);
      ierr = VecScatterCreate(schur->vec_G,is_G,schur->vec1_B,PETSC_NULL,&schur->G_to_B);CHKERRQ(ierr);
      ierr = ISDestroy(is_G);CHKERRQ(ierr);
      /* all global to local interior */
      ierr = ISGetIndices(schur->is_I,&idx);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(map_G,schur->n_I,idx,idx_G);CHKERRQ(ierr);
      ierr = ISRestoreIndices(schur->is_I,&idx);CHKERRQ(ierr);
      ierr = ISCreateGeneralWithArray(PETSC_COMM_SELF,schur->n_I,idx_G,&is_G);CHKERRQ(ierr);
      ierr = VecScatterCreate(schur->vec_G,is_G,schur->vec1_I,PETSC_NULL,&schur->G_to_I);CHKERRQ(ierr);
      ierr = ISDestroy(is_G);CHKERRQ(ierr);
      /* free auxiliar data */
      ierr = ISLocalToGlobalMappingDestroy(map_G);CHKERRQ(ierr);
      ierr = PetscFree(idx_G);CHKERRQ(ierr);
    }

    /* vector scatter from global interface (S) to local interface (B) */
    if(!schur->seq) { /* the following is only needed for the parallel case */
      PetscInt rstart,rend;
      IS       is_S;
      /* local index set for global interface (S) nodes in global, natural numbering */
      ierr = VecGetOwnershipRange(schur->vec1_S,&rstart,&rend);CHKERRQ(ierr);
      ierr = ISCreateStride(pc->comm,schur->n_S,rstart,1,&is_S);CHKERRQ(ierr);
      /* vector scatter from global interface (S) to local interface (B) nodes */
      ierr = VecScatterCreate(schur->vec1_S,is_S,schur->vec1_B,PETSC_NULL,&schur->S_to_B);CHKERRQ(ierr);
      ierr = ISDestroy(is_S);CHKERRQ(ierr);
    }

    ierr = ISDestroy(subdomain);CHKERRQ(ierr);
    ierr = ISDestroy(interface);CHKERRQ(ierr);
    ierr = ISDestroy(separator);CHKERRQ(ierr);
    
    /* create local subdomain submatrices */
    ierr = MatGetSubMatrix(schur->A,schur->is_I,schur->is_I,PETSC_DECIDE,MAT_INITIAL_MATRIX,&schur->A_II);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(schur->A,schur->is_I,schur->is_B,PETSC_DECIDE,MAT_INITIAL_MATRIX,&schur->A_IB);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(schur->A,schur->is_B,schur->is_I,PETSC_DECIDE,MAT_INITIAL_MATRIX,&schur->A_BI);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(schur->A,schur->is_B,schur->is_B,PETSC_DECIDE,MAT_INITIAL_MATRIX,&schur->A_BB);CHKERRQ(ierr);
    
    /* create global Schur complement operator */
    ierr = MatCreateShell(pc->comm,schur->n_S,schur->n_S,schur->N_S,schur->N_S,schur,&schur->mat_S);CHKERRQ(ierr);
    ierr = MatShellSetOperation(schur->mat_S,MATOP_MULT,(void(*)(void))MatMult_Schur);CHKERRQ(ierr);
    ierr = MatShellSetOperation(schur->mat_S,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Schur);CHKERRQ(ierr);
    ierr = MatShellSetOperation(schur->mat_S,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Schur);CHKERRQ(ierr);
    ierr = MatShellSetOperation(schur->mat_S,MATOP_VIEW,(void(*)(void))MatView_Schur);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)schur->mat_S,"MatGetDiagonalBlock_C",
					     "MatGetDiagonalBlock_Schur",MatGetDiagonalBlock_Schur);CHKERRQ(ierr);

    /* create, customize and setup local interior solver */
    ierr = KSPCreate(PETSC_COMM_SELF,&schur->ksp_I);CHKERRQ(ierr);
    ierr = KSPSetType(schur->ksp_I,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(schur->ksp_I,&sub_pc);CHKERRQ(ierr);
    ierr = PCSetType(sub_pc,PCLU);CHKERRQ(ierr);
    ierr = KSPSetOperators(schur->ksp_I,schur->A_II,schur->A_II,pc->flag);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(schur->ksp_I,prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(schur->ksp_I,"sub_local_");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(schur->ksp_I);CHKERRQ(ierr);
    /*ierr = KSPSetUp(schur->ksp_I);CHKERRQ(ierr);*/

    /* create, customize and setup global interface solver */
    ierr = KSPCreate(pc->comm,&schur->ksp_S);CHKERRQ(ierr);
    ierr = KSPGetPC(schur->ksp_S,&sub_pc);CHKERRQ(ierr);
    ierr = PCSetType(sub_pc,PCBJACOBI);CHKERRQ(ierr); /* XXX perhaps PCNONE? */
    ierr = KSPSetOperators(schur->ksp_S,schur->mat_S,schur->mat_S,pc->flag);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(schur->ksp_S,prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(schur->ksp_S,"sub_");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(schur->ksp_S);CHKERRQ(ierr);
    /*ierr = KSPSetUp(schur->ksp_S);CHKERRQ(ierr);*/

    /* log all internal objects */
    ierr = PetscLogObjectParent(pc,schur->is_I);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->is_B);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->vec1_I);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->vec2_I);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->vec3_I);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->vec1_B);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->vec2_B);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->vec3_B);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->G_to_I);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->G_to_B);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->A_II);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->A_IB);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->A_BI);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->A_BB);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->ksp_I);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->vec1_S);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->vec2_S);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->S_to_B);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->mat_S);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,schur->ksp_S);CHKERRQ(ierr);

  } else if (pc->flag == SAME_NONZERO_PATTERN) {

    /* update local diag and offdiag matrices */
    ierr = PCSchur_GetLocalMatrices(pc,pc->pmat,&schur->A,&schur->B,
				    PETSC_NULL,&schur->lvec,&schur->Mvctx);CHKERRQ(ierr);

    /* update local submatrices */
    ierr = MatGetSubMatrix(schur->A,schur->is_I,schur->is_I,PETSC_DECIDE,MAT_REUSE_MATRIX,&schur->A_II);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(schur->A,schur->is_I,schur->is_B,PETSC_DECIDE,MAT_REUSE_MATRIX,&schur->A_IB);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(schur->A,schur->is_B,schur->is_I,PETSC_DECIDE,MAT_REUSE_MATRIX,&schur->A_BI);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(schur->A,schur->is_B,schur->is_B,PETSC_DECIDE,MAT_REUSE_MATRIX,&schur->A_BB);CHKERRQ(ierr);

    /* update and setup local interior solver */
    ierr = KSPSetOperators(schur->ksp_I,schur->A_II,schur->A_II,pc->flag);CHKERRQ(ierr);
    /*ierr = KSPSetUp(schur->ksp_I);CHKERRQ(ierr);*/

    /* update and setup global interface solver */
    ierr = KSPSetOperators(schur->ksp_S,schur->mat_S,schur->mat_S,pc->flag);CHKERRQ(ierr);
    /*ierr = KSPSetUp(schur->ksp_S);CHKERRQ(ierr);*/

  } else {

    /* XXX rebuild everything, better way? */
    ierr = PCDestroy_Schur(pc);CHKERRQ(ierr);
    ierr = PCCreate_Schur(pc);CHKERRQ(ierr);
    pc->setupcalled = 0;
    ierr = PCSetUp_Schur(pc);CHKERRQ(ierr);
    pc->setupcalled = 2;

  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "PCSchur_Rhs"
static PetscErrorCode PCSchur_Rhs(PC pc,
				  Vec b,Vec b_I,Vec b_B,Vec b_S,
				  PetscTruth transposed)
{
  PC_Schur       *schur = (PC_Schur*)(pc->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* b_I, b_B  <-  b[I], b[B] */
  ierr = VecScatterBegin(schur->G_to_I,b,b_I,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (schur->G_to_I,b,b_I,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(schur->G_to_B,b,b_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (schur->G_to_B,b,b_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
  if (!transposed) { /* b_B  <-  b_B - A_BI * A_II^{-1} * b_I  */
    ierr = KSPSolve(schur->ksp_I,b_I,schur->vec1_I);CHKERRQ(ierr);
    ierr = MatMult(schur->A_BI,schur->vec1_I,schur->vec1_B);CHKERRQ(ierr);
    ierr = VecAXPY(b_B,-1,schur->vec1_B);CHKERRQ(ierr);
  } else {           /* b_B  <-  b_B - A_IB' * A_II^{-1}' * b_I  */
    ierr = KSPSolveTranspose(schur->ksp_I,b_I,schur->vec1_I);CHKERRQ(ierr);
    ierr = MatMultTranspose(schur->A_IB,schur->vec1_I,schur->vec1_B);CHKERRQ(ierr);
    ierr = VecAXPY(b_B,-1,schur->vec1_B);CHKERRQ(ierr);
  }

  if (schur->seq) {   /* b_S  <-  b_B*/
    ierr = VecCopy(b_B, b_S);CHKERRQ(ierr);
  } else {            /* b_S[S]  <-  b_S[S] + b_B */
    ierr = VecSet(b_S,0);CHKERRQ(ierr);
    ierr = VecScatterBegin(schur->S_to_B,b_B,b_S,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (schur->S_to_B,b_B,b_S,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef  __FUNCT__  
#define __FUNCT__ "PCSchur_Solve"
static PetscErrorCode PCSchur_Solve(PC pc,
				    Vec b_S,Vec x_S,
				    PetscTruth transposed)
{
  PC_Schur       *schur = (PC_Schur*)(pc->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  if (!transposed) { /* x_S <-  S^{-1} b_S */
    ierr = KSPSolve(schur->ksp_S,b_S,x_S);CHKERRQ(ierr);
  } else {           /* x_S <-  S'^{-1} b_S */
    ierr = KSPSolveTranspose(schur->ksp_S,b_S,x_S);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef  __FUNCT__  
#define __FUNCT__ "PCSchur_Solution"
static PetscErrorCode PCSchur_Solution(PC pc,
				       Vec x_S,Vec b_I,Vec x_I,Vec x_B,Vec x,
				       PetscTruth transposed)
{
  PC_Schur       *schur = (PC_Schur*)(pc->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (schur->seq) { /* x_B <- x_S */
    ierr = VecCopy(x_S, x_B);CHKERRQ(ierr);
  } else {          /* x_B <- x_S[B] */
    ierr = VecScatterBegin(schur->S_to_B,x_S,x_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (schur->S_to_B,x_S,x_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  if (!transposed) { /* x_I <- inv(A_II) * (b_I - A_IB * x_B) */
    ierr = MatMult(schur->A_IB,x_B,x_I);CHKERRQ(ierr);
    ierr = VecAXPY(b_I,-1,x_I);CHKERRQ(ierr);
    ierr = KSPSolve(schur->ksp_I,b_I,x_I);CHKERRQ(ierr);
  } else {           /* x_I <- A_II^{-1}' * (b_I - A_BI' * x_B) */
    ierr = MatMultTranspose(schur->A_BI,x_B,x_I);CHKERRQ(ierr);
    ierr = VecAXPY(b_I,-1,x_I);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(schur->ksp_I,b_I,x_I);CHKERRQ(ierr);
  }

  /* x[I], x[B]  <- x_I, x_B  */
  ierr = VecScatterBegin(schur->G_to_I,x_I,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (schur->G_to_I,x_I,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(schur->G_to_B,x_B,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (schur->G_to_B,x_B,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Schur"
static PetscErrorCode PCSchur_Apply(PC pc,Vec r,Vec z,PetscTruth transposed)
{
  PC_Schur       *schur = (PC_Schur*)(pc->data);
  Vec            x_I    = schur->vec1_I;
  Vec            b_I    = schur->vec3_I;
  Vec            x_B    = schur->vec1_B;
  Vec            b_B    = schur->vec2_B;
  Vec            x_S    = schur->vec1_S;
  Vec            b_S    = schur->vec2_S;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* build global rhs */
  ierr = PCSchur_Rhs(pc,r,b_I,b_B,b_S,transposed);CHKERRQ(ierr);
  /* solve global Schur problem */
  ierr = PCSchur_Solve(pc,b_S,x_S,transposed);CHKERRQ(ierr);
  /* build global solution */
  ierr = PCSchur_Solution(pc,x_S,b_I,x_I,x_B,z,transposed);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */

/*
   PCApply_Schur - Applies the Schur complement preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  r - input vector (global)

   Output Parameter:
.  z - output vector (global)

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_Schur"
static PetscErrorCode PCApply_Schur(PC pc,Vec r,Vec z)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PCSchur_Apply(pc,r,z,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*
   PCApplyTranspose_Schur - Applies the transpose of Schur complement 
   preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  r - input vector (global)

   Output Parameter:
.  z - output vector (global)

   Application Interface Routine: PCApplyTranspose()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_Schur"
static PetscErrorCode PCApplyTranspose_Schur(PC pc,Vec r,Vec z)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PCSchur_Apply(pc,r,z,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*
  PCView_Schur - 

   Input Parameter:
.  pc - the preconditioner context
.  viewer - the viewer context

  Application Interface Routine: PCView()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCView_Schur"
static PetscErrorCode PCView_Schur(PC pc,PetscViewer viewer)
{
  PC_Schur       *schur = (PC_Schur*)pc->data;
  PetscTruth     iascii;
  PetscMPIInt    size,rank;
  PetscViewer    sviewer;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (!iascii) {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PC Schur",((PetscObject)viewer)->type_name);
    PetscFunctionReturn(PETSC_ERR_SUP);
  }

  ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
  
  /* options */
  if (schur->blocks != PETSC_DECIDE)
    {ierr = PetscViewerASCIIPrintf(viewer,"  Schur: requested local blocks = %D\n",schur->blocks);CHKERRQ(ierr);}
  if (schur->ccsize != PETSC_DECIDE)
    {ierr = PetscViewerASCIIPrintf(viewer,"  Schur: requested local ccsize = %D\n",schur->ccsize);CHKERRQ(ierr);}
  /* subdomain sub-partitioning statistics */
  if (schur->sp_stats && pc->setupcalled) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Sub-partitioning in each processor\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ----------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] neqs = %D, blocks = %D, block sizes (min-max) = %D-%D, nseps = %D\n",
					      rank,schur->n,schur->sp_count,schur->sp_minsz,schur->sp_maxsz,schur->sp_nseps);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  /* global interface solver */
  ierr = PetscViewerASCIIPrintf(viewer,"  KSP and PC for global interface problem\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ---------------------------------------\n");CHKERRQ(ierr);
  if (schur->ksp_S) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(schur->ksp_S,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"  no yet created\n");CHKERRQ(ierr);
  }
  /* local interior solver */
  ierr = PetscViewerASCIIPrintf(viewer,"  KSP and PC for local interior problem\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  -------------------------------------\n");CHKERRQ(ierr);
  ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
  if (!rank) {
    if (schur->ksp_I) {
      ierr = PetscViewerASCIIPushTab(sviewer);CHKERRQ(ierr);
      ierr = KSPView(schur->ksp_I,sviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(sviewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  no yet created\n");CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSchurGetSubKSP(PC,PetscInt*,KSP**);
PETSC_EXTERN_CXX_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCSchurGetSubKSP_Schur"
PetscErrorCode PETSCKSP_DLLEXPORT PCSchurGetSubKSP_Schur(PC pc,PetscInt *n,KSP **ksp)
{
  PC_Schur       *schur = (PC_Schur*)pc->data;
  PetscFunctionBegin;
  if (pc->setupcalled) {
    schur->subksp[0] = schur->ksp_S;
    schur->subksp[1] = schur->ksp_I;
    if (n)   *n   = 2;
    if (ksp) *ksp = schur->subksp;
  } else {
    if (n)   *n   = 0;
    if (ksp) *ksp = PETSC_NULL;
    SETERRQ(PETSC_ERR_ORDER,"Need to call PCSetUP() on PC (or KSPSetUp() on the outer KSP object) before calling this");
    PetscFunctionReturn(PETSC_ERR_ORDER);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCSchurGetSubKSP"
/*@C
   PCSchurGetSubKSP - 

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  n - the number of KSP contexts
-  ksp - the array of KSP contexts

   Note:  
   After PCSchurGetSubKSP() the array of KSPes is not to be freed

   You must call KSPSetUp() before calling PCSchurGetSubKSP().

   Level: advanced

.keywords: PC, KSP

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCSchurGetSubKSP(PC pc,PetscInt *n,KSP *ksp[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt*,KSP **);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCSchurGetSubKSP_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,ksp);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot get subsolvers for this type of PC");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONG);
  }
 PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*
   PCDestroy_Schur - Destroys the private context for the Schur preconditioner
   that was created with PCCreate_Schur().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Schur"
static PetscErrorCode PCDestroy_Schur(PC pc)
{
  PC_Schur       *schur = (PC_Schur*)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Free internal objects */

  if (schur->vec_G) {ierr = VecDestroy(schur->vec_G);CHKERRQ(ierr);}
  
  if (schur->is_I)   {ierr = ISDestroy(schur->is_I);CHKERRQ(ierr);}
  if (schur->is_B)   {ierr = ISDestroy(schur->is_B);CHKERRQ(ierr);}
  if (schur->vec1_I) {ierr = VecDestroy(schur->vec1_I);CHKERRQ(ierr);}
  if (schur->vec2_I) {ierr = VecDestroy(schur->vec2_I);CHKERRQ(ierr);}
  if (schur->vec3_I) {ierr = VecDestroy(schur->vec3_I);CHKERRQ(ierr);}
  if (schur->vec1_B) {ierr = VecDestroy(schur->vec1_B);CHKERRQ(ierr);}
  if (schur->vec2_B) {ierr = VecDestroy(schur->vec2_B);CHKERRQ(ierr);}
  if (schur->vec3_B) {ierr = VecDestroy(schur->vec3_B);CHKERRQ(ierr);}
  if (schur->G_to_I) {ierr = VecScatterDestroy(schur->G_to_I);CHKERRQ(ierr);}
  if (schur->G_to_B) {ierr = VecScatterDestroy(schur->G_to_B);CHKERRQ(ierr);}

  if (schur->A_II)  {ierr = MatDestroy(schur->A_II);CHKERRQ(ierr);}
  if (schur->A_IB)  {ierr = MatDestroy(schur->A_IB);CHKERRQ(ierr);}
  if (schur->A_BI)  {ierr = MatDestroy(schur->A_BI);CHKERRQ(ierr);}
  if (schur->A_BB)  {ierr = MatDestroy(schur->A_BB);CHKERRQ(ierr);}
  if (schur->ksp_I) {ierr = KSPDestroy(schur->ksp_I);CHKERRQ(ierr);}

  if (schur->vec1_S) {ierr = VecDestroy(schur->vec1_S);CHKERRQ(ierr);}
  if (schur->vec2_S) {ierr = VecDestroy(schur->vec2_S);CHKERRQ(ierr);}
  if (schur->S_to_B) {ierr = VecScatterDestroy(schur->S_to_B);CHKERRQ(ierr);}
  if (schur->mat_S)  {ierr = MatDestroy(schur->mat_S);CHKERRQ(ierr);}
  if (schur->ksp_S)  {ierr = KSPDestroy(schur->ksp_S);CHKERRQ(ierr);}

  /* Free the private data structure */
  ierr = PetscFree(schur);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCSchurGetSubKSP_C","",0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*MC
   PCSCHUR - Schur complement preconditioner.

   Options Database Keys:
+  -pc_schur_local_blocks - XXX explain this
.  -pc_schur_local_ccsize - XXX explain this
.  -pc_schur_strip_layers - XXX explain this
.  -pc_schur_print_stats - XXX explain this
.  -sub_[ksp,pc]_ - Options for global interface solver (default types: KSPGMRES with PCJACOBI)
.  -sub_strip_[ksp,pc]_ - Options for global strip solver (default types: KSPRICHARDSON with PCJACOBI)
-  -sub_local_[ksp,pc]_ - Options for local interior solver (default types: KSPPREONLY with PCLU)

   Level: intermediate

   Concepts: Iterative Substructuring methods.

   Notes: 
   The matrix used with this preconditioner must be of type MATIS in more than one processor.

   Notes: 
   Usually this will compute an aproximate solution at interface nodes
   in one iteration and does not need a Krylov method (i.e. you can
   use -ksp_type preonly, or KSPSetType(ksp,KSPPREONLY) for the Krylov
   method. If you have an good initial guess, you should use a
   Richardson method with only one iteration (i.e. you should use
   -ksp_type richardon -ksp_max_it 1, or use the following
.vb
       KSPSetType(ksp,KSPRICHARDSON);
       KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1);
.ve

   Contributed by Lisandro Dalcin <dalcinl at gmail dot com>

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,  MATIS
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Schur"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Schur(PC pc)
{
  PC_Schur       *schur;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscNew(PC_Schur,&schur);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_Schur));CHKERRQ(ierr);

  pc->data  = (void*)schur;

  schur->seq = PETSC_FALSE;

  schur->blocks = PETSC_DECIDE;
  schur->ccsize = PETSC_DECIDE;
  
  schur->sp_stats = PETSC_FALSE;


  schur->A = 0;
  schur->B = 0;

  schur->lvec  = 0;
  schur->Mvctx = 0;
  schur->vec_G = 0;

  schur->n   = 0;
  schur->n_I = 0;
  schur->n_B = 0;
  schur->is_I = 0;
  schur->is_B = 0;

  schur->vec1_I = 0;
  schur->vec2_I = 0;
  schur->vec3_I = 0;
  schur->vec1_B = 0;
  schur->vec2_B = 0;
  schur->vec3_B = 0;

  schur->G_to_I = 0;
  schur->G_to_B = 0;

  schur->A_II  = 0;
  schur->A_IB  = 0;
  schur->A_BI  = 0;
  schur->A_BB  = 0;
  schur->ksp_I = 0;

  schur->n_S    = 0;
  schur->N_S    = 0;
  schur->vec1_S = 0;
  schur->vec2_S = 0;
  schur->S_to_B = 0;
  schur->mat_S  = 0;
  schur->ksp_S  = 0;
  
  pc->ops->setfromoptions      = PCSetFromOptions_Schur;
  pc->ops->setup               = PCSetUp_Schur;
  pc->ops->apply               = PCApply_Schur;
  pc->ops->applytranspose      = PCApplyTranspose_Schur;;
  pc->ops->view                = PCView_Schur;
  pc->ops->destroy             = PCDestroy_Schur;
  pc->ops->presolve            = 0;
  pc->ops->postsolve           = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,
					   "PCSchurGetSubKSP_C","PCSchurGetSubKSP_Schur",
					   PCSchurGetSubKSP_Schur);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
#undef VecScatterBegin
#undef VecScatterEnd
#undef MatGetRowIJ
#undef MatRestoreRowIJ
#endif
