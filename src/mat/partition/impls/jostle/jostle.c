#define PETSCMAT_DLL

#include "../src/mat/impls/adj/mpi/mpiadj.h"       /*I "petscmat.h" I*/

#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef PETSC_HAVE_STDLIB_H
#include <stdlib.h>
#endif

EXTERN_C_BEGIN

#include "jostle.h"
/* this function is not declared in 'jostle.h' */
extern void pjostle_comm(MPI_Comm * comm);

EXTERN_C_END 

typedef struct {
    int output;
    int coarse_seq;
    int nbvtxcoarsed;           /* number of vertices for the coarse graph */
    char *mesg_log;
} MatPartitioning_Jostle;

#define SIZE_LOG 10000          /* size of buffer for msg_log */

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningApply_Jostle"
static PetscErrorCode MatPartitioningApply_Jostle(MatPartitioning part, IS * partitioning)
{
    PetscErrorCode ierr;
    int  size, rank, i;
    Mat mat = part->adj, matMPI;
    Mat_MPIAdj *adj = (Mat_MPIAdj *) mat->data;
    MatPartitioning_Jostle *jostle_struct = (MatPartitioning_Jostle *) part->data;
    PetscTruth flg;
#ifdef PETSC_HAVE_UNISTD_H
    int fd_stdout, fd_pipe[2], count,err;
#endif

    PetscFunctionBegin;

    /* check that the number of partitions is equal to the number of processors */
    ierr = MPI_Comm_rank(((PetscObject)mat)->comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject)mat)->comm, &size);CHKERRQ(ierr);
    if (part->n != size) SETERRQ(PETSC_ERR_SUP, "Supports exactly one domain per processor");

    /* convert adjacency matrix to MPIAdj if needed*/
    ierr = PetscTypeCompare((PetscObject) mat, MATMPIADJ, &flg);CHKERRQ(ierr);
    if (!flg) {
        ierr = MatConvert(mat, MATMPIADJ, MAT_INITIAL_MATRIX, &matMPI);CHKERRQ(ierr);
    } else {
        matMPI = mat;
    }

    adj = (Mat_MPIAdj *) matMPI->data;  /* adj contains adjacency graph */
    {
        /* definition of Jostle library arguments */
        int nnodes = matMPI->M; /* number of vertices in full graph */
        int offset = 0;         /* 0 for C array indexing */
        int core = matMPI->m;
        int halo = 0;           /* obsolete with contiguous format */
        int *index_jostle;      /* contribution of each processor */
        int nparts = part->n;
        int *part_wt = NULL;

        int *partition;         /* set number of each vtx (length n) */
        int *degree;            /* degree for each core nodes */
        int *edges = adj->j;
        int *node_wt = NULL;    /* nodes weights */
        int *edge_wt = NULL;    /* edges weights */
        double *coords = NULL;  /* not used (cf jostle documentation) */

        int local_nedges = adj->nz;
        int dimension = 0;      /* not used */
        int output_level = jostle_struct->output;
        char env_str[256];

        /* allocate index_jostle */
        ierr = PetscMalloc(nparts * sizeof(int), &index_jostle);CHKERRQ(ierr);

        /* compute number of core nodes for each one */
        for (i = 0; i < nparts - 1; i++)
            index_jostle[i] = adj->rowners[i + 1] - adj->rowners[i];
        index_jostle[nparts - 1] = nnodes - adj->rowners[nparts - 1];

        /* allocate the partition vector */
        ierr = PetscMalloc(core * sizeof(int), &partition);CHKERRQ(ierr); 

        /* build the degree vector and the local_nedges value */
        ierr = PetscMalloc(core * sizeof(int), &degree);CHKERRQ(ierr);
        for (i = 0; i < core; i++)
            degree[i] = adj->i[i + 1] - adj->i[i];

        /* library call */
        pjostle_init(&size, &rank);
        pjostle_comm(&((PetscObject)matMPI)->comm);
        jostle_env("format = contiguous");
        jostle_env("timer = off");

        sprintf(env_str, "threshold = %d", jostle_struct->nbvtxcoarsed);
        jostle_env(env_str);

        if (jostle_struct->coarse_seq)
          jostle_env("matching = local");

        /* redirect output */
#ifdef PETSC_HAVE_UNISTD_H
        fd_stdout = dup(1);
        pipe(fd_pipe);
        close(1);
        dup2(fd_pipe[1], 1);
#endif

        pjostle(&nnodes, &offset, &core, &halo, index_jostle, degree, node_wt,
            partition, &local_nedges, edges, edge_wt, &nparts,
            part_wt, &output_level, &dimension, coords);

        printf("Jostle Partitioner statistics\ncut : %d, balance : %f, runtime : %f, mem used : %d\n",
            jostle_cut(), jostle_bal(), jostle_tim(), jostle_mem());

#ifdef PETSC_HAVE_UNISTD_H
        ierr = PetscMalloc(SIZE_LOG * sizeof(char), &(jostle_struct->mesg_log));CHKERRQ(ierr);
        err = fflush(stdout);
        if (err) SETERRQ(PETSC_ERR_SYS,"fflush() failed on stdout");    
        count = read(fd_pipe[0], jostle_struct->mesg_log, (SIZE_LOG - 1) * sizeof(char));
        if (count < 0)
            count = 0;
        jostle_struct->mesg_log[count] = 0;
        close(1);
        dup2(fd_stdout, 1);
        close(fd_stdout);
        close(fd_pipe[0]);
        close(fd_pipe[1]);
#endif

        /* We free the memory used by jostle */
        ierr = PetscFree(index_jostle);CHKERRQ(ierr);
        ierr = PetscFree(degree);CHKERRQ(ierr);

        /* Creation of the index set */
        ierr = ISCreateGeneral(((PetscObject)part)->comm, mat->m, partition, partitioning);CHKERRQ(ierr);

        if (matMPI != mat) {
            ierr = MatDestroy(matMPI);CHKERRQ(ierr);
        }

        ierr = PetscFree(partition);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningView_Jostle"
PetscErrorCode MatPartitioningView_Jostle(MatPartitioning part, PetscViewer viewer)
{
  MatPartitioning_Jostle *jostle_struct = (MatPartitioning_Jostle *) part->data;
  PetscErrorCode         ierr;
  PetscTruth             iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    if (jostle_struct->mesg_log) {
      ierr = PetscViewerASCIIPrintf(viewer, "%s\n", jostle_struct->mesg_log);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Viewer type %s not supported for this Jostle partitioner",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningJostleSetCoarseLevel"
/*@
    MatPartitioningJostleSetCoarseLevel - Set the coarse level 
    
  Input Parameter:
.  part - the partitioning context
.  level - the coarse level in range [0.0,1.0]

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningJostleSetCoarseLevel(MatPartitioning part, PetscReal level)
{
    MatPartitioning_Jostle *jostle_struct = (MatPartitioning_Jostle *) part->data;

    PetscFunctionBegin;

    if (level < 0.0 || level > 1.0) {
        SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,
            "Jostle: level of coarsening out of range [0.0-1.0]");
    } else
        jostle_struct->nbvtxcoarsed = (int)(part->adj->N * level);

    if (jostle_struct->nbvtxcoarsed < 20)
        jostle_struct->nbvtxcoarsed = 20;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningJostleSetCoarseSequential"
/*@
     MatPartitioningJostleSetCoarseSequential - Use the sequential code to 
         do the partitioning of the coarse grid.

  Input Parameter:
.  part - the partitioning context

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningJostleSetCoarseSequential(MatPartitioning part)
{
    MatPartitioning_Jostle *jostle_struct =
        (MatPartitioning_Jostle *) part->data;
    PetscFunctionBegin;
    jostle_struct->coarse_seq = 1;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningSetFromOptions_Jostle"
PetscErrorCode MatPartitioningSetFromOptions_Jostle(MatPartitioning part)
{
    PetscErrorCode ierr;
    PetscTruth     flag = PETSC_FALSE;
    PetscReal      level;

    PetscFunctionBegin;
    ierr = PetscOptionsHead("Set Jostle partitioning options");CHKERRQ(ierr);

    ierr = PetscOptionsReal("-mat_partitioning_jostle_coarse_level","Coarse level", "MatPartitioningJostleSetCoarseLevel", 0, &level, &flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatPartitioningJostleSetCoarseLevel(part, level);CHKERRQ(ierr);
    }

    flag = PETSC_FALSE;
    ierr = PetscOptionsTruth("-mat_partitioning_jostle_coarse_sequential","Use sequential coarse partitioner","MatPartitioningJostleSetCoarseSequential",flag,&flag,PETSC_NULL);CHKERRQ(ierr);
    if (flag) {
      ierr = MatPartitioningJostleSetCoarseSequential(part);CHKERRQ(ierr);
    }

    ierr = PetscOptionsTail();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningDestroy_Jostle"
PetscErrorCode MatPartitioningDestroy_Jostle(MatPartitioning part)
{
    MatPartitioning_Jostle *jostle_struct = (MatPartitioning_Jostle *) part->data;
    PetscErrorCode         ierr;

    PetscFunctionBegin;
    ierr = PetscFree(jostle_struct->mesg_log);CHKERRQ(ierr);
    ierr = PetscFree(jostle_struct);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*MC
   MAT_PARTITIONING_JOSTLE - Creates a partitioning context via the external package Jostle.

   Collective on MPI_Comm

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:
+  -mat_partitioning_jostle_coarse_level <0>: Coarse level (MatPartitioningJostleSetCoarseLevel)
-  -mat_partitioning_jostle_coarse_sequential: Use sequential coarse partitioner (MatPartitioningJostleSetCoarseSequential)

   Level: beginner

   Notes: See http://www.gre.ac.uk/~c.walshaw/jostle/

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningCreate_Jostle"
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningCreate_Jostle(MatPartitioning part)
{
    PetscErrorCode ierr;
    MatPartitioning_Jostle *jostle_struct;

    PetscFunctionBegin;
    ierr = PetscNewLog(part,MatPartitioning_Jostle, &jostle_struct);CHKERRQ(ierr);
    part->data = (void*) jostle_struct;

    jostle_struct->nbvtxcoarsed = 20;
    jostle_struct->output = 0;
    jostle_struct->coarse_seq = 0;
    jostle_struct->mesg_log = NULL;

    part->ops->apply = MatPartitioningApply_Jostle;
    part->ops->view = MatPartitioningView_Jostle;
    part->ops->destroy = MatPartitioningDestroy_Jostle;
    part->ops->setfromoptions = MatPartitioningSetFromOptions_Jostle;

    PetscFunctionReturn(0);
}

EXTERN_C_END
