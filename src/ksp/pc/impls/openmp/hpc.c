
#include <petsc-private/pcimpl.h>     /*I "petscpc.h" I*/
#include <petscksp.h>

typedef struct {
  MatStructure flag;               /* pc->flag */
  PetscInt     setupcalled;        /* pc->setupcalled */
  PetscInt     n;
  MPI_Comm     comm;                 /* local world used by this preconditioner */
  KSP          ksp;                  /* actual solver used across local world */
  Mat          mat;                  /* matrix in local world */
  Mat          gmat;                 /* matrix known only to process 0 in the local world */
  Vec          x,y,xdummy,ydummy;
  VecScatter   scatter;
  PetscBool    nonzero_guess;
} PC_HMPI;


#undef __FUNCT__
#define __FUNCT__ "PCView_HMPI_MP"
/*
    Would like to have this simply call PCView() on the inner PC. The problem is
  that the outer comm does not live on the inside so cannot do this. Instead
  handle the special case when the viewer is stdout, construct a new one just
  for this call.
*/

static PetscErrorCode PCView_HMPI_MP(MPI_Comm comm,void *ctx)
{
  PC_HMPI      *red = (PC_HMPI*)ctx;
  PetscErrorCode ierr;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);         /* this is bogus in general */
  ierr = KSPView(red->ksp,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_HMPI"
static PetscErrorCode PCView_HMPI(PC pc,PetscViewer viewer)
{
  PC_HMPI      *red = (PC_HMPI*)pc->data;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(red->comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Size of solver nodes %d\n",size);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Parallel sub-solver given next\n",size);CHKERRQ(ierr);
    /* should only make the next call if the viewer is associated with stdout */
    ierr = PetscHMPIRun(red->comm,PCView_HMPI_MP,red);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatDistribute_MPIAIJ(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);

#undef __FUNCT__
#define __FUNCT__ "PCApply_HMPI_1"
static PetscErrorCode PCApply_HMPI_1(PC pc,Vec x,Vec y)
{
  PC_HMPI      *red = (PC_HMPI*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetInitialGuessNonzero(red->ksp,pc->nonzero_guess);CHKERRQ(ierr);
  ierr = KSPSolve(red->ksp,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_HMPI_MP"
static PetscErrorCode PCSetUp_HMPI_MP(MPI_Comm comm,void *ctx)
{
  PC_HMPI      *red = (PC_HMPI*)ctx;
  PetscErrorCode ierr;
  PetscInt       m;
  MatReuse       scal;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  red->comm = comm;
  ierr = MPI_Bcast(&red->setupcalled,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&red->flag,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  if (!red->setupcalled) {
    /* setup vector communication */
    ierr = MPI_Bcast(&red->n,1,MPIU_INT,0,comm);CHKERRQ(ierr);
    ierr = VecCreateMPI(comm,PETSC_DECIDE,red->n,&red->x);CHKERRQ(ierr);
    ierr = VecCreateMPI(comm,PETSC_DECIDE,red->n,&red->y);CHKERRQ(ierr);
    ierr = VecScatterCreateToZero(red->x,&red->scatter,&red->xdummy);CHKERRQ(ierr);
    ierr = VecDuplicate(red->xdummy,&red->ydummy);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = VecDestroy(&red->xdummy);CHKERRQ(ierr);
      ierr = VecDestroy(&red->ydummy);CHKERRQ(ierr);
    }
    scal = MAT_INITIAL_MATRIX;
  } else {
    if (red->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr = MatDestroy(&red->mat);CHKERRQ(ierr);
      scal = MAT_INITIAL_MATRIX;
      CHKMEMQ;
    } else {
      scal = MAT_REUSE_MATRIX;
    }
  }

  /* copy matrix out onto processes */
  ierr = VecGetLocalSize(red->x,&m);CHKERRQ(ierr);
  ierr = MatDistribute_MPIAIJ(comm,red->gmat,m,scal,&red->mat);CHKERRQ(ierr);
  if (!red->setupcalled) {
    /* create the solver */
    ierr = KSPCreate(comm,&red->ksp);CHKERRQ(ierr);
    /* would like to set proper tablevel for KSP, but do not have direct access to parent pc */
    ierr = KSPSetOptionsPrefix(red->ksp,"hmpi_");CHKERRQ(ierr); /* should actually append with global pc prefix */
    ierr = KSPSetOperators(red->ksp,red->mat,red->mat,red->flag);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(red->ksp);CHKERRQ(ierr);
  } else {
    ierr = KSPSetOperators(red->ksp,red->mat,red->mat,red->flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_HMPI"
static PetscErrorCode PCSetUp_HMPI(PC pc)
{
  PC_HMPI      *red = (PC_HMPI*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  red->gmat        = pc->mat;
  red->flag        = pc->flag;
  red->setupcalled = pc->setupcalled;

  ierr = MPI_Comm_size(red->comm,&size);CHKERRQ(ierr);
  if (size == 1) {  /* special case where copy of matrix is not needed */
    if (!red->setupcalled) {
      /* create the solver */
      ierr = KSPCreate(((PetscObject)pc)->comm,&red->ksp);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)red->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(red->ksp,"hmpi_");CHKERRQ(ierr); /* should actually append with global pc prefix */
      ierr = KSPSetOperators(red->ksp,red->gmat,red->gmat,red->flag);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(red->ksp);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(red->ksp,red->gmat,red->gmat,red->flag);CHKERRQ(ierr);
    }
    pc->ops->apply = PCApply_HMPI_1;
    PetscFunctionReturn(0);
  } else {
    ierr = MatGetSize(pc->mat,&red->n,PETSC_IGNORE);CHKERRQ(ierr);
    ierr = PetscHMPIRun(red->comm,PCSetUp_HMPI_MP,red);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_HMPI_MP"
static PetscErrorCode PCApply_HMPI_MP(MPI_Comm comm,void *ctx)
{
  PC_HMPI      *red = (PC_HMPI*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(red->scatter,red->xdummy,red->x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->scatter,red->xdummy,red->x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MPI_Bcast(&red->nonzero_guess,1,MPIU_INT,0,red->comm);CHKERRQ(ierr);
  if (red->nonzero_guess) {
    ierr = VecScatterBegin(red->scatter,red->ydummy,red->y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(red->scatter,red->ydummy,red->y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  ierr = KSPSetInitialGuessNonzero(red->ksp,red->nonzero_guess);CHKERRQ(ierr);

  ierr = KSPSolve(red->ksp,red->x,red->y);CHKERRQ(ierr);

  ierr = VecScatterBegin(red->scatter,red->y,red->ydummy,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->scatter,red->y,red->ydummy,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_HMPI"
static PetscErrorCode PCApply_HMPI(PC pc,Vec x,Vec y)
{
  PC_HMPI      *red = (PC_HMPI*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  red->xdummy        = x;
  red->ydummy        = y;
  red->nonzero_guess = pc->nonzero_guess;
  ierr = PetscHMPIRun(red->comm,PCApply_HMPI_MP,red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_HMPI_MP"
static PetscErrorCode PCDestroy_HMPI_MP(MPI_Comm comm,void *ctx)
{
  PC_HMPI      *red = (PC_HMPI*)ctx;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterDestroy(&red->scatter);CHKERRQ(ierr);
  ierr = VecDestroy(&red->x);CHKERRQ(ierr);
  ierr = VecDestroy(&red->y);CHKERRQ(ierr);
  ierr = KSPDestroy(&red->ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&red->mat);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) {
    ierr = VecDestroy(&red->xdummy);CHKERRQ(ierr);
    ierr = VecDestroy(&red->ydummy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_HMPI"
static PetscErrorCode PCDestroy_HMPI(PC pc)
{
  PC_HMPI      *red = (PC_HMPI*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHMPIRun(red->comm,PCDestroy_HMPI_MP,red);CHKERRQ(ierr);
  ierr = PetscHMPIFree(red->comm,red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_HMPI"
static PetscErrorCode PCSetFromOptions_HMPI(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------------------*/
/*MC
     PCHMPI - Runs a preconditioner for a single process matrix across several MPI processes

$     This will usually be run with -pc_type hmpi -ksp_type preonly
$     solver options are set with -hmpi_ksp_... and -hmpi_pc_... for example
$     -hmpi_ksp_type cg would use cg as the Krylov method or -hmpi_ksp_monitor or
$     -hmpi_pc_type hypre -hmpi_pc_hypre_type boomeramg

       Always run with -ksp_view (or -snes_view) to see what solver is actually being used.

       Currently the solver options INSIDE the HMPI preconditioner can ONLY be set via the
      options database.

   Level: intermediate

   See PetscHMPIMerge() and PetscHMPISpawn() for two ways to start up MPI for use with this preconditioner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types)

M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_HMPI"
PetscErrorCode  PCCreate_HMPI(PC pc)
{
  PetscErrorCode ierr;
  PC_HMPI      *red;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr      = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_SIZ,"HMPI preconditioner only works for sequential solves");
  if (!PETSC_COMM_LOCAL_WORLD) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PETSc not initialized for PCMPI see the manual pages for PetscHMPISpawn() and PetscHMPIMerge()");
  /* caste the struct length to a PetscInt for easier MPI calls */

  ierr      = PetscHMPIMalloc(PETSC_COMM_LOCAL_WORLD,(PetscInt)sizeof(PC_HMPI),(void**)&red);CHKERRQ(ierr);
  red->comm = PETSC_COMM_LOCAL_WORLD;
  pc->data  = (void*) red;

  pc->ops->apply          = PCApply_HMPI;
  pc->ops->destroy        = PCDestroy_HMPI;
  pc->ops->setfromoptions = PCSetFromOptions_HMPI;
  pc->ops->setup          = PCSetUp_HMPI;
  pc->ops->view           = PCView_HMPI;
  PetscFunctionReturn(0);
}
EXTERN_C_END
