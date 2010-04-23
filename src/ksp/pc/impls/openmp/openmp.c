#define PETSCKSP_DLL

#include "private/pcimpl.h"     /*I "petscpc.h" I*/
#include "petscksp.h"

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
  PetscTruth   nonzero_guess; 
} PC_OpenMP;


#undef __FUNCT__  
#define __FUNCT__ "PCView_OpenMP_OpenMP"
/*
    Would like to have this simply call PCView() on the inner PC. The problem is
  that the outter comm does not live on the inside so cannot do this. Instead 
  handle the special case when the viewer is stdout, construct a new one just
  for this call.
*/

static PetscErrorCode PCView_OpenMP_MP(MPI_Comm comm,void *ctx)
{
  PC_OpenMP      *red = (PC_OpenMP*)ctx;
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
#define __FUNCT__ "PCView_OpenMP"
static PetscErrorCode PCView_OpenMP(PC pc,PetscViewer viewer)
{
  PC_OpenMP      *red = (PC_OpenMP*)pc->data;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(red->comm,&size);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Size of solver nodes %d\n",size);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Parallel sub-solver given next\n",size);CHKERRQ(ierr);
    /* should only make the next call if the viewer is associated with stdout */
    ierr = PetscOpenMPRun(red->comm,PCView_OpenMP_MP,red);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatDistribute_MPIAIJ(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);

#undef __FUNCT__  
#define __FUNCT__ "PCApply_OpenMP_1"
static PetscErrorCode PCApply_OpenMP_1(PC pc,Vec x,Vec y)
{
  PC_OpenMP      *red = (PC_OpenMP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetInitialGuessNonzero(red->ksp,pc->nonzero_guess);CHKERRQ(ierr);
  ierr = KSPSolve(red->ksp,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_OpenMP_MP"
static PetscErrorCode PCSetUp_OpenMP_MP(MPI_Comm comm,void *ctx)
{
  PC_OpenMP      *red = (PC_OpenMP*)ctx;
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
      ierr = VecDestroy(red->xdummy);CHKERRQ(ierr);
      ierr = VecDestroy(red->ydummy);CHKERRQ(ierr);
    }
    scal = MAT_INITIAL_MATRIX;
  } else {
    if (red->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr = MatDestroy(red->mat);CHKERRQ(ierr);
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
    ierr = KSPSetOptionsPrefix(red->ksp,"openmp_");CHKERRQ(ierr); /* should actually append with global pc prefix */
    ierr = KSPSetOperators(red->ksp,red->mat,red->mat,red->flag);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(red->ksp);CHKERRQ(ierr);
  } else {
    ierr = KSPSetOperators(red->ksp,red->mat,red->mat,red->flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_OpenMP"
static PetscErrorCode PCSetUp_OpenMP(PC pc)
{
  PC_OpenMP      *red = (PC_OpenMP*)pc->data;
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
      ierr = KSPSetOptionsPrefix(red->ksp,"openmp_");CHKERRQ(ierr); /* should actually append with global pc prefix */
      ierr = KSPSetOperators(red->ksp,red->gmat,red->gmat,red->flag);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(red->ksp);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(red->ksp,red->gmat,red->gmat,red->flag);CHKERRQ(ierr);
    }
    pc->ops->apply = PCApply_OpenMP_1;
    PetscFunctionReturn(0);
  } else {
    ierr = MatGetSize(pc->mat,&red->n,PETSC_IGNORE);CHKERRQ(ierr); 
    ierr = PetscOpenMPRun(red->comm,PCSetUp_OpenMP_MP,red);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_OpenMP_MP"
static PetscErrorCode PCApply_OpenMP_MP(MPI_Comm comm,void *ctx)
{
  PC_OpenMP      *red = (PC_OpenMP*)ctx;
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
#define __FUNCT__ "PCApply_OpenMP"
static PetscErrorCode PCApply_OpenMP(PC pc,Vec x,Vec y)
{
  PC_OpenMP      *red = (PC_OpenMP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  red->xdummy        = x;
  red->ydummy        = y;
  red->nonzero_guess = pc->nonzero_guess;
  ierr = PetscOpenMPRun(red->comm,PCApply_OpenMP_MP,red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_OpenMP_MP"
static PetscErrorCode PCDestroy_OpenMP_MP(MPI_Comm comm,void *ctx)
{
  PC_OpenMP      *red = (PC_OpenMP*)ctx;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (red->scatter) {ierr = VecScatterDestroy(red->scatter);CHKERRQ(ierr);}
  if (red->x) {ierr = VecDestroy(red->x);CHKERRQ(ierr);}
  if (red->y) {ierr = VecDestroy(red->y);CHKERRQ(ierr);}
  if (red->ksp) {ierr = KSPDestroy(red->ksp);CHKERRQ(ierr);}
  if (red->mat) {ierr = MatDestroy(red->mat);CHKERRQ(ierr);}
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) {
    if (red->xdummy) {ierr = VecDestroy(red->xdummy);CHKERRQ(ierr);}
    if (red->ydummy) {ierr = VecDestroy(red->ydummy);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_OpenMP"
static PetscErrorCode PCDestroy_OpenMP(PC pc)
{
  PC_OpenMP      *red = (PC_OpenMP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOpenMPRun(red->comm,PCDestroy_OpenMP_MP,red);CHKERRQ(ierr);
  ierr = PetscOpenMPFree(red->comm,red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_OpenMP"
static PetscErrorCode PCSetFromOptions_OpenMP(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------------------*/
/*MC
     PCOPENMP - Runs a preconditioner for a single process matrix across several MPI processes

$     This will usually be run with -pc_type openmp -ksp_type preonly
$     solver options are set with -openmp_ksp_... and -openmp_pc_... for example
$     -openmp_ksp_type cg would use cg as the Krylov method or -openmp_ksp_monitor or
$     -openmp_pc_type hypre -openmp_pc_hypre_type boomeramg

       Always run with -ksp_view (or -snes_view) to see what solver is actually being used.

       Currently the solver options INSIDE the OpenMP preconditioner can ONLY be set via the
      options database.

   Level: intermediate

   See PetscOpenMPMerge() and PetscOpenMPSpawn() for two ways to start up MPI for use with this preconditioner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types)

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_OpenMP"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_OpenMP(PC pc)
{
  PetscErrorCode ierr;
  PC_OpenMP      *red;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr      = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_SIZ,"OpenMP preconditioner only works for sequential solves");
  /* caste the struct length to a PetscInt for easier MPI calls */

  ierr      = PetscOpenMPMalloc(PETSC_COMM_LOCAL_WORLD,(PetscInt)sizeof(PC_OpenMP),(void**)&red);CHKERRQ(ierr);
  red->comm = PETSC_COMM_LOCAL_WORLD;
  pc->data  = (void*) red;

  pc->ops->apply          = PCApply_OpenMP;
  pc->ops->destroy        = PCDestroy_OpenMP;
  pc->ops->setfromoptions = PCSetFromOptions_OpenMP;
  pc->ops->setup          = PCSetUp_OpenMP;
  pc->ops->view           = PCView_OpenMP;
  PetscFunctionReturn(0);
}
EXTERN_C_END
