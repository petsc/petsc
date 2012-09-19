

static char help[] = "Demonstrates PetscHMPIMerge() usage\n\n";

#include <petscmat.h>
#include <petscksp.h>

typedef struct {
  MPI_Comm   comm;
  Mat        A;
  Vec        x,y;      /* contains the vector values spread across all the processes */
  Vec        xr,yr;    /* contains the vector values on the master processes, all other processes have zero length */
  VecScatter sct;
} MyMultCtx;

#undef __FUNCT__
#define __FUNCT__ "MyMult"
/*
    This is called on ALL processess, master and worker
*/
PetscErrorCode MyMult(MPI_Comm comm,MyMultCtx *ctx,void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSynchronizedPrintf(ctx->comm,"Doing multiply\n");
  ierr = PetscSynchronizedFlush(ctx->comm);CHKERRQ(ierr);
  /* moves data that lives only on master processes to all processes */
  ierr = VecScatterBegin(ctx->sct,ctx->xr,ctx->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->sct,ctx->xr,ctx->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,ctx->x,ctx->y);CHKERRQ(ierr);
  /* moves data that lives on all processes to master processes */
  ierr = VecScatterBegin(ctx->sct,ctx->y,ctx->yr,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->sct,ctx->y,ctx->yr,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MySubsolver"
/*
    This is called only on the master processes
*/
PetscErrorCode MySubsolver(MyMultCtx *ctx)
{
  PetscErrorCode ierr;
  void           *subctx;

  PetscFunctionBegin;
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"MySubsolver\n");
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  /* allocates memory on each process, both masters and workers */
  ierr = PetscHMPIMalloc(PETSC_COMM_LOCAL_WORLD,sizeof(int),&subctx);CHKERRQ(ierr);
  /* runs MyMult() function on each process, both masters and workers */
  ierr = PetscHMPIRunCtx(PETSC_COMM_LOCAL_WORLD,(PetscErrorCode (*)(MPI_Comm,void*,void *))MyMult,subctx);CHKERRQ(ierr);
  ierr = PetscHMPIFree(PETSC_COMM_LOCAL_WORLD,subctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscMPIInt    rank,size,nodesize = 1;
  MyMultCtx      ctx;
  const PetscInt *ns; /* length of vector ctx.x on all process */
  PetscInt       i,rstart,n = 0;   /* length of vector ctx.xr on this process */
  IS             is;

  PetscInitialize(&argc,&args,(char *)0,help);
  ctx.comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nodesize",&nodesize,PETSC_NULL);CHKERRQ(ierr);
  if (size % nodesize) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"MPI_COMM_WORLD size must be divisible by nodesize");

  /* Read matrix */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&ctx.A);CHKERRQ(ierr);
  ierr = MatSetType(ctx.A,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatLoad(ctx.A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Create work vectors for matrix-vector product */
  ierr = MatGetVecs(ctx.A,&ctx.x,&ctx.y);CHKERRQ(ierr);
  ierr = VecGetOwnershipRanges(ctx.x,&ns);CHKERRQ(ierr);
  if (!(rank % nodesize)) { /* I am master process; I will own all vector elements on all my worker processes*/
    for (i=0; i<nodesize; i++) n += ns[rank+i+1] - ns[rank+i];
  }
  ierr = VecCreateMPI(MPI_COMM_WORLD,n,PETSC_DETERMINE,&ctx.xr);CHKERRQ(ierr);
  ierr = VecDuplicate(ctx.xr,&ctx.yr);CHKERRQ(ierr);
  /* create scatter from ctx.xr to ctx.x vector */
  ierr = VecGetOwnershipRange(ctx.x,&rstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,ns[rank],rstart,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(ctx.xr,is,ctx.x,is,&ctx.sct);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /*
     The master nodes call the function MySubsolver() while the worker nodes wait for requests to call functions
     These requests are triggered by the calls from the masters on PetscHMPIRunCtx()
  */
  ierr = PetscHMPIMerge(nodesize,(PetscErrorCode (*)(void*))MySubsolver,&ctx);CHKERRQ(ierr);

  ierr = PetscHMPIFinalize();CHKERRQ(ierr);
  ierr = MatDestroy(&ctx.A);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.x);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.y);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.xr);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.yr);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx.sct);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

