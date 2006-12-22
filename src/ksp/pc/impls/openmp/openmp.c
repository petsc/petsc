#define PETSCKSP_DLL

#include "private/pcimpl.h"     /*I "petscpc.h" I*/
#include "petscksp.h"

typedef struct {
  PetscInt   n;
  MPI_Comm   comm;                 /* local world used by this preconditioner */
  PC         pc;                   /* actual preconditioner used across local world */
  Mat        mat;                  /* matrix in local world */
  Mat        gmat;                 /* matrix known only to process 0 in the local world */
  Vec        x,y,xdummy,ydummy;
  VecScatter scatter;
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
  ierr = PCView(red->pc,viewer);CHKERRQ(ierr);
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

#include "include/private/matimpl.h"        /*I "petscmat.h" I*/
#include "private/vecimpl.h" 
#include "src/mat/impls/aij/mpi/mpiaij.h"   /*I "petscmat.h" I*/
#include "src/mat/impls/aij/seq/aij.h"      /*I "petscmat.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "MatDistribute_MPIAIJ"
/*
    Distributes a SeqAIJ matrix across a set of processes. Code stolen from
    MatLoad_MPIAIJ(). Horrible lack of reuse. Should be a routine for each matrix type.

    Only for square matrices
*/
static PetscErrorCode MatDistribute_MPIAIJ(MPI_Comm comm,Mat gmat,PetscInt m,Mat *inmat)
{
  PetscMPIInt    rank,size;
  PetscInt       *rowners,*dlens,*olens,i,rstart,rend,j,jj,nz,*gmataj,cnt,row;
  PetscErrorCode ierr;
  Mat            mat;
  Mat_SeqAIJ     *gmata;
  PetscMPIInt    tag;
  MPI_Status     status;
  PetscTruth     aij;
  PetscScalar    *gmataa;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscTypeCompare((PetscObject)gmat,MATSEQAIJ,&aij);CHKERRQ(ierr);
    if (!aij) SETERRQ1(PETSC_ERR_SUP,"Currently no support for input matrix of type %s\n",gmat->type_name);
  }
  ierr = MatCreate(comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m,m,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(PetscInt),&rowners);CHKERRQ(ierr);
  ierr = PetscMalloc2(m,PetscInt,&dlens,m,PetscInt,&olens);CHKERRQ(ierr);
  ierr = MPI_Allgather(&m,1,MPIU_INT,rowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 
  ierr   = PetscObjectGetNewTag((PetscObject)mat,&tag);CHKERRQ(ierr);
  if (!rank) {
    gmata = (Mat_SeqAIJ*) gmat->data;
    /* send row lengths to all processors */
    for (i=0; i<m; i++) dlens[i] = gmata->ilen[i];
    for (i=1; i<size; i++) {
      ierr = MPI_Send(gmata->ilen + rowners[i],rowners[i+1]-rowners[i],MPIU_INT,i,tag,comm);CHKERRQ(ierr);
    }
    /* determine number diagonal and off-diagonal counts */
    ierr = PetscMemzero(olens,m*sizeof(PetscInt));CHKERRQ(ierr);
    jj = 0;
    for (i=0; i<m; i++) {
      for (j=0; j<dlens[i]; j++) {
        if (gmata->j[jj] < rstart || gmata->j[jj] >= rend) olens[i]++;
        jj++;
      }
    }
    /* send column indices to other processes */
    for (i=1; i<size; i++) {
      nz   = gmata->i[rowners[i+1]]-gmata->i[rowners[i]];
      ierr = MPI_Send(&nz,1,MPIU_INT,i,tag,comm);CHKERRQ(ierr);
      ierr = MPI_Send(gmata->j + gmata->i[rowners[i]],gmata->i[rowners[i+1]]-gmata->i[rowners[i]],MPIU_INT,i,tag,comm);CHKERRQ(ierr);
    }
    /* send numerical values to other processes */
    for (i=1; i<size; i++) {
      nz   = gmata->i[rowners[i+1]]-gmata->i[rowners[i]];
      ierr = MPI_Send(gmata->a + gmata->i[rowners[i]],gmata->i[rowners[i+1]]-gmata->i[rowners[i]],MPIU_SCALAR,i,tag,comm);CHKERRQ(ierr);
    }
    gmataa = gmata->a;
    gmataj = gmata->j;

  } else {
    /* receive row lengths */
    ierr = MPI_Recv(dlens,m,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
    /* receive column indices */
    ierr = MPI_Recv(&nz,1,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = PetscMalloc2(nz,PetscScalar,&gmataa,nz,PetscInt,&gmataj);CHKERRQ(ierr);
    ierr = MPI_Recv(gmataj,nz,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
    /* determine number diagonal and off-diagonal counts */
    ierr = PetscMemzero(olens,m*sizeof(PetscInt));CHKERRQ(ierr);
    jj = 0;
    for (i=0; i<m; i++) {
      for (j=0; j<dlens[i]; j++) {
        if (gmataj[jj] < rstart || gmataj[jj] >= rend) olens[i]++;
        jj++;
      }
    }
    /* receive numerical values */
  ierr = PetscMemzero(gmataa,nz*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MPI_Recv(gmataa,nz,MPIU_SCALAR,0,tag,comm,&status);CHKERRQ(ierr);
  }
  /* set preallocation */
  for (i=0; i<m; i++) {
    dlens[i] -= olens[i];
  }
  ierr = MatSeqAIJSetPreallocation(mat,0,dlens);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dlens,0,olens);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    dlens[i] += olens[i];
  }
  cnt  = 0;
  for (i=0; i<m; i++) {
    row  = rstart + i;
    ierr = MatSetValues(mat,1,&row,dlens[i],gmataj+cnt,gmataa+cnt,INSERT_VALUES);CHKERRQ(ierr);
    cnt += dlens[i];
  }
  if (rank) {
    ierr = PetscFree2(gmataa,gmataj);CHKERRQ(ierr);
  }
  ierr = PetscFree2(dlens,olens);CHKERRQ(ierr);
  ierr = PetscFree(rowners);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  CHKMEMQ;
  *inmat = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_OpenMP_MP"
static PetscErrorCode PCSetUp_OpenMP_MP(MPI_Comm comm,void *ctx)
{
  PC_OpenMP      *red = (PC_OpenMP*)ctx;
  PetscErrorCode ierr;
  PetscInt       m;

  PetscFunctionBegin;
  /* setup vector communication */
  ierr = MPI_Bcast(&red->n,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,PETSC_DECIDE,red->n,&red->x);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,PETSC_DECIDE,red->n,&red->y);CHKERRQ(ierr);
  ierr = VecScatterCreateToZero(red->x,&red->scatter,&red->xdummy);CHKERRQ(ierr);
  ierr = VecDuplicate(red->xdummy,&red->ydummy);CHKERRQ(ierr);

  /* copy matrix out onto processes */
  ierr = VecGetLocalSize(red->x,&m);CHKERRQ(ierr);
  ierr = MatDistribute_MPIAIJ(comm,red->gmat,m,&red->mat);CHKERRQ(ierr);

  /* create the solver */
  ierr = PCCreate(comm,&red->pc);CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(red->pc,"openmp_");CHKERRQ(ierr); /* should actually append with global pc prefix */
  ierr = PCSetType(red->pc,PCKSP);CHKERRQ(ierr);
  ierr = PCSetOperators(red->pc,red->mat,red->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PCSetFromOptions(red->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_OpenMP"
static PetscErrorCode PCSetUp_OpenMP(PC pc)
{
  PC_OpenMP      *red = (PC_OpenMP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  red->gmat = pc->mat;
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
  ierr = VecScatterBegin(red->xdummy,red->x,INSERT_VALUES,SCATTER_REVERSE,red->scatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->xdummy,red->x,INSERT_VALUES,SCATTER_REVERSE,red->scatter);CHKERRQ(ierr);

  ierr = PCApply(red->pc,red->x,red->y);CHKERRQ(ierr);

  ierr = VecScatterBegin(red->y,red->ydummy,INSERT_VALUES,SCATTER_FORWARD,red->scatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->y,red->ydummy,INSERT_VALUES,SCATTER_FORWARD,red->scatter);CHKERRQ(ierr);
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
  if (red->scatter) {ierr = VecScatterDestroy(red->scatter);CHKERRQ(ierr);}
  if (red->x) {ierr = VecDestroy(red->x);CHKERRQ(ierr);}
  if (red->y) {ierr = VecDestroy(red->y);CHKERRQ(ierr);}
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

$     This will usually be run with -pc_type openmp -ksp_type gmres -openmp_pc_type ksp then
$     solver options are set with -openmp_ksp_ksp_... and -openmp_ksp_pc_... for example
$     -openmp_ksp_ksp_type cg would use cg as the Krylov method or -openmp_ksp_ksp_monitor

       Always run with -ksp_view (or -snes_view) to see what solver is actually being used.

   Level: intermediate

   See PetscOpenMPMerge() and PetscOpenMPSpawn() for two ways to start up MPI for use with this preconditioner

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
