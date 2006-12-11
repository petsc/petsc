#define PETSCKSP_DLL

#include "private/pcimpl.h"     /*I "petscpc.h" I*/
#include "petscksp.h"

typedef struct {
  PetscInt   n;
  MPI_Comm   comm;                 /* local world used by this preconditioner */
  PC         pc;                   /* actual preconditioner used across local world */
  Vec        x,y,xdummy,ydummy;
  VecScatter scatter;
} PC_OpenMP;


#undef __FUNCT__  
#define __FUNCT__ "PCView_OpenMP"
static PetscErrorCode PCView_OpenMP(PC pc,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#include "include/private/matimpl.h"        /*I "petscmat.h" I*/
#include "private/vecimpl.h" 
#include "src/mat/impls/aij/mpi/mpiaij.h"   /*I "petscmat.h" I*/
#include "src/mat/impls/aij/seq/aij.h"      /*I "petscmat.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_OpenMP_MP"
static PetscErrorCode PCSetUp_OpenMP_MP(MPI_Comm comm,void *ctx)
{
  PC_OpenMP      *red = (PC_OpenMP*)ctx;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Bcast(&red->n,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,PETSC_DECIDE,red->n,&red->x);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,PETSC_DECIDE,red->n,&red->y);CHKERRQ(ierr);
  ierr = VecScatterCreateToZero(red->x,&red->scatter,&red->xdummy);CHKERRQ(ierr);
  ierr = VecDuplicate(red->xdummy,&red->ydummy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_OpenMP"
static PetscErrorCode PCSetUp_OpenMP(PC pc)
{
  PC_OpenMP      *red = (PC_OpenMP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(pc->mat,&red->n,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = PetscOpenMPRun(red->comm,PCSetUp_OpenMP_MP,red);CHKERRQ(ierr);
  ierr = VecDestroy(red->xdummy);CHKERRQ(ierr);
  ierr = VecDestroy(red->ydummy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_OpenMP_MP"
static PetscErrorCode PCApply_OpenMP_MP(MPI_Comm comm,void *ctx)
{
  PC_OpenMP      *red = (PC_OpenMP*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(red->xdummy,red->x,INSERT_VALUES,SCATTER_FORWARD,red->scatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->xdummy,red->x,INSERT_VALUES,SCATTER_FORWARD,red->scatter);CHKERRQ(ierr);

  ierr = VecScatterBegin(red->y,red->ydummy,INSERT_VALUES,SCATTER_REVERSE,red->scatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->y,red->ydummy,INSERT_VALUES,SCATTER_REVERSE,red->scatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_OpenMP"
static PetscErrorCode PCApply_OpenMP(PC pc,Vec x,Vec y)
{
  PC_OpenMP      *red = (PC_OpenMP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  red->xdummy = x;
  red->ydummy = y;
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
  ierr = VecScatterDestroy(red->scatter);CHKERRQ(ierr);
  ierr = VecDestroy(red->x);CHKERRQ(ierr);
  ierr = VecDestroy(red->y);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) {
    ierr = VecDestroy(red->xdummy);CHKERRQ(ierr);
    ierr = VecDestroy(red->ydummy);CHKERRQ(ierr);
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

     Options for the openmp preconditioners can be set with -openmp_pc_xxx

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types)

M*/
extern MPI_Comm PETSC_COMM_LOCAL_WORLD;

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_OpenMP"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_OpenMP(PC pc)
{
  PetscErrorCode ierr;
  PC_OpenMP      *red;
  const char     *prefix;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr      = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_SIZ,"OpenMP preconditioner only works for sequential solves");

  ierr      = PetscOpenMPNew(PETSC_COMM_LOCAL_WORLD,sizeof(PC_OpenMP),(void**)&red);CHKERRQ(ierr);
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
