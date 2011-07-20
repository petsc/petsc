/* 
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */
#include <private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>                    /*I "petscpcmg.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

/* Private context for the GAMG preconditioner */
typedef struct gamg_TAG {
  PetscInt       m_dim;
  PetscInt       m_Nlevels;
  PetscInt       m_data_sz;
  PetscReal     *m_data; /* blocked vector of vertex data on fine grid (coordinates) */
} PC_GAMG;

/* -----------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PCReset_GAMG"
PetscErrorCode PCReset_GAMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->m_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetCoordinates_GAMG

   Input Parameter:
   .  pc - the preconditioner context
*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCSetCoordinates_GAMG"
PetscErrorCode PCSetCoordinates_GAMG(PC pc, const int ndm,PetscReal *coords )
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscErrorCode ierr;  
  PetscInt       bs, my0, tt;
  Mat            mat = pc->pmat; 
  PetscInt       arrsz;

  PetscFunctionBegin;
  ierr  = MatGetBlockSize( mat, &bs );               CHKERRQ( ierr );
  ierr  = MatGetOwnershipRange( mat, &my0, &tt ); CHKERRQ(ierr);
  arrsz = (tt-my0)/bs*ndm;

  // put coordinates
  if (!pc_gamg->m_data || (pc_gamg->m_data_sz != arrsz)) {
    ierr = PetscFree( pc_gamg->m_data );  CHKERRQ(ierr);
    ierr = PetscMalloc(arrsz*sizeof(double), &pc_gamg->m_data ); CHKERRQ(ierr);
  }

  /* copy data in */
  for(tt=0;tt<arrsz;tt++){
    pc_gamg->m_data[tt] = coords[tt];
  }
  pc_gamg->m_data_sz = arrsz;
  pc_gamg->m_dim = ndm;

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
/*
   createCrsOp

   Input Parameter:
   . Amat - matrix on this fine level
   . P_out - prolongation operator to the next level
   . Acrs - coarse matrix that is created
*/
#undef __FUNCT__
#define __FUNCT__ "createCrsOp"
PetscErrorCode createCrsOp( Mat Amat, Mat P_inout, Mat *Acrs )
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mat H;
  ierr = MatPtAP( Amat, P_inout, MAT_INITIAL_MATRIX, 2.0, &H); CHKERRQ(ierr);

  /* need to repartition H and move colums of P accordingly */
  

  *Acrs = H;  

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_GAMG - Prepares for the use of the GAMG preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
extern PetscErrorCode PCSetFromOptions_MG(PC);
extern PetscErrorCode PCReset_MG(PC);
extern PetscErrorCode createProlongation( Mat, PetscReal [], const PetscInt,
                                          Mat *, PetscReal **, PetscBool *a_isOK );
#undef __FUNCT__
#define __FUNCT__ "PCSetUp_GAMG"
PetscErrorCode PCSetUp_GAMG( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  Mat              Amat = pc->mat, Pmat = pc->pmat;
  PetscBool        isSeq, isMPI;
  PetscInt         fine_level, level, level1, M, N, bs, lidx;
  MPI_Comm         wcomm = ((PetscObject)pc)->comm;
  PetscMPIInt      mype,npe;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  if (pc->setupcalled){
    /* no state data in GAMG to destroy (now) */
    ierr = PCReset_MG(pc); CHKERRQ(ierr);
  }
  if (!pc_gamg->m_data) SETERRQ(wcomm,PETSC_ERR_SUP,"PCSetUp_GAMG called before PCSetCoordinates");
  /* setup special features of PCGAMG */
  ierr = PetscTypeCompare((PetscObject)Amat, MATSEQAIJ, &isSeq);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)Amat, MATMPIAIJ, &isMPI);CHKERRQ(ierr);
  if (isMPI) {
  } else if (isSeq) {
  } else SETERRQ1(wcomm,PETSC_ERR_ARG_WRONG, "Matrix type '%s' cannot be used with GAMG. GAMG can only handle AIJ matrices.",((PetscObject)Amat)->type_name);
  
  /* GAMG requires input of fine-grid matrix. It determines nlevels. */
  ierr = MatGetSize( Amat, &M, &N );CHKERRQ(ierr);
  ierr = MatGetBlockSize( Amat, &bs ); CHKERRQ(ierr);
  if(bs!=1) SETERRQ1(wcomm,PETSC_ERR_ARG_WRONG, "GAMG only supports scalar prblems bs = '%d'.",bs);

  /* Get A_i and R_i */
#define GAMG_MAXLEVELS 10
  Mat Aarr[GAMG_MAXLEVELS], Rarr[GAMG_MAXLEVELS];  PetscReal *coarse_crds = 0, *crds = pc_gamg->m_data;
  PetscBool isOK;
  for (level=0, Aarr[0] = Pmat; level < GAMG_MAXLEVELS-1; level++ ){
    ierr = MatGetSize( Aarr[level], &M, &N );CHKERRQ(ierr);
    if( M < npe*10 ) { /* hard wire this for now */
      break;
    }
    level1 = level + 1;
    PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s make level %d N=%d\n",0,__FUNCT__,level+1,N);
    ierr = createProlongation( Aarr[level], crds, pc_gamg->m_dim,
                               &Rarr[level1], &coarse_crds, &isOK );
    CHKERRQ(ierr);
    ierr = PetscFree( crds ); CHKERRQ( ierr );
    crds = coarse_crds;
    if(level==0) Aarr[0] = Amat; /* use Pmat for finest level setup, but use mat for solver */
    if( isOK ) {
      ierr = createCrsOp( Aarr[level], Rarr[level1], &Aarr[level1] ); CHKERRQ(ierr);
    }
    else{
      break;
    }
  }
  ierr = PetscFree( coarse_crds ); CHKERRQ( ierr );
PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s %d levels\n",0,__FUNCT__,level + 1);
  pc_gamg->m_data = 0; /* destroyed coordinate data */
  pc_gamg->m_Nlevels = level + 1;
  fine_level = level;
  ierr = PCMGSetLevels(pc,pc_gamg->m_Nlevels,PETSC_NULL);CHKERRQ(ierr);

  /* set default smoothers */
  PetscReal emax = 2.0, emin;
  for (level=1; level<=fine_level; level++){
    KSP smoother; PC subpc;
    ierr = PCMGGetSmoother(pc,level,&smoother);CHKERRQ(ierr);
    ierr = KSPSetType(smoother,KSPCHEBYCHEV);CHKERRQ(ierr);
    emin = emax/10.0; /* fix!!! */
    ierr = KSPGetPC(smoother,&subpc);CHKERRQ(ierr);
    ierr = PCSetType(subpc,PCJACOBI);CHKERRQ(ierr);
    ierr = KSPChebychevSetEigenvalues(smoother, emax, emin);CHKERRQ(ierr); /* need auto !!!!*/
  }
  ierr = PCSetFromOptions_MG(pc); CHKERRQ(ierr); /* should be called in PCSetFromOptions_GAMG(), but cannot be called prior to PCMGSetLevels() */
  {
    PetscBool galerkin;
    ierr = PCMGGetGalerkin( pc,  &galerkin); CHKERRQ(ierr);
    if(galerkin){
      SETERRQ(wcomm,PETSC_ERR_ARG_WRONG, "GAMG does galerkin manually so it must not be used in PC_MG.");
    }
  }
  /* create coarse level and the interpolation between the levels */
  for (level=0,lidx=pc_gamg->m_Nlevels-1; level<fine_level; level++,lidx--){
    level1 = level + 1;
    /* PetscInt MM,NN; */
    /* ierr = MatGetSize( Rarr[lidx], &MM, &NN );CHKERRQ(ierr); */
    /* PetscPrintf(PETSC_COMM_WORLD,"%s Set P(%d,%d) on level %d (%d)\n",__FUNCT__,MM,NN,level1,lidx); */
    ierr = PCMGSetInterpolation(pc,level1,Rarr[lidx]);CHKERRQ(ierr);
    if(!PETSC_TRUE) {
      PetscViewer        viewer;
      ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, "Rmat.m", &viewer);  CHKERRQ(ierr);
      ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
      ierr = MatView(Rarr[lidx],viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy( &viewer );
    }
    ierr = MatDestroy( &Rarr[lidx] );  CHKERRQ(ierr);
    {
      KSP smoother;
      ierr = PCMGGetSmoother(pc,level,&smoother); CHKERRQ(ierr);
      ierr = KSPSetOperators( smoother, Aarr[lidx], Aarr[lidx], DIFFERENT_NONZERO_PATTERN );
      CHKERRQ(ierr);
      ierr = MatDestroy( &Aarr[lidx] );  CHKERRQ(ierr);
    }
  }
  { /* fine level (no P) */
    KSP smoother;
    ierr = PCMGGetSmoother(pc,fine_level,&smoother); CHKERRQ(ierr);
    ierr = KSPSetOperators( smoother, Aarr[0], Aarr[0], DIFFERENT_NONZERO_PATTERN );
    CHKERRQ(ierr);
  }

  /* setupcalled is set to 0 so that MG is setup from scratch */
  pc->setupcalled = 0;
  ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCDestroy_GAMG - Destroys the private context for the GAMG preconditioner
   that was created with PCCreate_GAMG().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_GAMG"
PetscErrorCode PCDestroy_GAMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg= (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PCReset_GAMG(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc_gamg);CHKERRQ(ierr);
  ierr = PCDestroy_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_GAMG"
PetscErrorCode PCSetFromOptions_GAMG(PC pc)
{
  /* PetscErrorCode  ierr; */
  /* PC_MG           *mg = (PC_MG*)pc->data; */
  /* PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx; */
  /* MPI_Comm        comm = ((PetscObject)pc)->comm; */

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 PCCreate_GAMG - Creates a GAMG preconditioner context, PC_GAMG

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()

  */
 /* MC
     PCGAMG - Use algebraic multigrid preconditioning. This preconditioner requires you provide
       fine grid discretization matrix and coordinates on the fine grid.

   Options Database Key:
   Multigrid options(inherited)
+  -pc_mg_cycles <1>: 1 for V cycle, 2 for W-cycle (MGSetCycles)
.  -pc_mg_smoothup <1>: Number of post-smoothing steps (MGSetNumberSmoothUp)
.  -pc_mg_smoothdown <1>: Number of pre-smoothing steps (MGSetNumberSmoothDown)
   -pc_mg_type <multiplicative>: (one of) additive multiplicative full cascade kascade
   GAMG options:

   Level: intermediate
  Concepts: multigrid

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType,
           PCMGSetLevels(), PCMGGetLevels(), PCMGSetType(), MPSetCycles(), PCMGSetNumberSmoothDown(),
           PCMGSetNumberSmoothUp(), PCMGGetCoarseSolve(), PCMGSetResidual(), PCMGSetInterpolation(),
           PCMGSetRestriction(), PCMGGetSmoother(), PCMGGetSmootherUp(), PCMGGetSmootherDown(),
           PCMGSetCyclesOnLevel(), PCMGSetRhs(), PCMGSetX(), PCMGSetR()
M */

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_GAMG"
PetscErrorCode  PCCreate_GAMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_GAMG         *pc_gamg;
  PC_MG           *mg;

  PetscFunctionBegin;
  /* PCGAMG is an inherited class of PCMG. Initialize pc as PCMG */
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr); /* calls PCCreate_MG() and MGCreate_Private() */
  ierr = PetscObjectChangeTypeName((PetscObject)pc,PCGAMG);CHKERRQ(ierr);

  /* create a supporting struct and attach it to pc */
  ierr = PetscNewLog(pc,PC_GAMG,&pc_gamg);CHKERRQ(ierr);
  mg = (PC_MG*)pc->data;
  mg->innerctx = pc_gamg;

  pc_gamg->m_Nlevels    = -1;

  /* overwrite the pointers of PCMG by the functions of PCGAMG */
  pc->ops->setfromoptions = PCSetFromOptions_GAMG;
  pc->ops->setup          = PCSetUp_GAMG;
  pc->ops->reset          = PCReset_GAMG;
  pc->ops->destroy        = PCDestroy_GAMG;

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCSetCoordinates_C",
					    "PCSetCoordinates_GAMG",
					    PCSetCoordinates_GAMG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
