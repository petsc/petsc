#define PETSCMAT_DLL

#include "../src/mat/impls/adj/mpi/mpiadj.h"       /*I "petscmat.h" I*/

#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef PETSC_HAVE_STDLIB_H
#include <stdlib.h>
#endif

/* 
   Currently using Party-1.99
*/
EXTERN_C_BEGIN
#include "party_lib.h"
EXTERN_C_END 

typedef struct {
    char redm[15];
    char redo[15];
    int rec;
    int output;
    char global_method[15];     /* global method */
    char local_method[15];      /* local method */
    int nbvtxcoarsed;           /* number of vertices for the coarse graph */
    char *mesg_log;
} MatPartitioning_Party;

#define SIZE_LOG 10000          /* size of buffer for msg_log */

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningApply_Party"
static PetscErrorCode MatPartitioningApply_Party(MatPartitioning part, IS * partitioning)
{
    PetscErrorCode ierr;
    int  *locals, *parttab = NULL, rank, size;
    Mat mat = part->adj, matMPI, matSeq;
    int nb_locals;              
    Mat_MPIAdj *adj = (Mat_MPIAdj *) mat->data;
    MatPartitioning_Party *party = (MatPartitioning_Party *) part->data;
    PetscTruth flg;
#ifdef PETSC_HAVE_UNISTD_H
    int fd_stdout, fd_pipe[2], count,err;
#endif

    PetscFunctionBegin;
    /* check if the matrix is sequential, use MatGetSubMatrices if necessary */
    ierr = PetscTypeCompare((PetscObject) mat, MATMPIADJ, &flg);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject)mat)->comm, &size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)part)->comm, &rank);CHKERRQ(ierr);
    if (size > 1) {
        int M, N;
        IS isrow, iscol;
        Mat *A;

        if (flg) SETERRQ(PETSC_ERR_SUP,"Distributed matrix format MPIAdj is not supported for sequential partitioners");
        ierr = PetscPrintf(((PetscObject)part)->comm,"Converting distributed matrix to sequential: this could be a performance loss\n");CHKERRQ(ierr);
        ierr = MatGetSize(mat, &M, &N);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF, M, 0, 1, &isrow);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF, N, 0, 1, &iscol);CHKERRQ(ierr);
        ierr = MatGetSubMatrices(mat, 1, &isrow, &iscol, MAT_INITIAL_MATRIX, &A);CHKERRQ(ierr);
        ierr = ISDestroy(isrow);CHKERRQ(ierr);
        ierr = ISDestroy(iscol);CHKERRQ(ierr);
        matSeq = *A;
        ierr   = PetscFree(A);CHKERRQ(ierr);
    } else {
        matSeq = mat;
    }
    /* check for the input format that is supported only for a MPIADJ type 
       and set it to matMPI */

    if (!flg) {
        ierr = MatConvert(matSeq, MATMPIADJ, MAT_INITIAL_MATRIX, &matMPI);CHKERRQ(ierr);
    } else {
        matMPI = matSeq;
    }

    adj = (Mat_MPIAdj *) matMPI->data;  /* finaly adj contains adjacency graph */
    {
        /* Party library arguments definition */
        int n = mat->rmap->N;         /* number of vertices in full graph */
        int *edge_p = adj->i;   /* start of edge list for each vertex */
        int *edge = adj->j;     /* edge list data */
        int *vertex_w = NULL;   /* weights for all vertices */
        int *edge_w = NULL;     /* weights for all edges */
        float *x = NULL, *y = NULL, *z = NULL;  /* coordinates for inertial method */
        int p = part->n;        /* number of parts to create */
        int *part_party;        /* set number of each vtx (length n) */
        int cutsize;            /* number of edge cut */
        char *global = party->global_method;    /* global partitioning algorithm */
        char *local = party->local_method;      /* local partitioning algorithm */
        int redl = party->nbvtxcoarsed; /* how many vertices to coarsen down to? */
        char *redm = party->redm;
        char *redo = party->redo;
        int rec = party->rec;
        int output = party->output;

        ierr = PetscMalloc((mat->rmap->N) * sizeof(int), &part_party);CHKERRQ(ierr);

        /* redirect output to buffer party->mesg_log */
#ifdef PETSC_HAVE_UNISTD_H
        fd_stdout = dup(1);
        pipe(fd_pipe);
        close(1);
        dup2(fd_pipe[1], 1);
        ierr = PetscMalloc(SIZE_LOG * sizeof(char), &(party->mesg_log));CHKERRQ(ierr);
#endif

        /* library call */
        party_lib_times_start();
        ierr = party_lib(n, vertex_w, x, y, z, edge_p, edge, edge_w,
            p, part_party, &cutsize, redl, redm, redo,
            global, local, rec, output);

        party_lib_times_output(output);
        part_info(n, vertex_w, edge_p, edge, edge_w, p, part_party, output);

#ifdef PETSC_HAVE_UNISTD_H
        err = fflush(stdout);
        if (err) SETERRQ(PETSC_ERR_SYS,"fflush() failed on stdout");    
        count =
            read(fd_pipe[0], party->mesg_log, (SIZE_LOG - 1) * sizeof(char));
        if (count < 0)
            count = 0;
        party->mesg_log[count] = 0;
        close(1);
        dup2(fd_stdout, 1);
        close(fd_stdout);
        close(fd_pipe[0]);
        close(fd_pipe[1]);
#endif
        /* if in the call we got an error, we say it */
        if (ierr) SETERRQ(PETSC_ERR_LIB, party->mesg_log);
        parttab = part_party;
    }

    /* Creation of the index set */
    ierr = MPI_Comm_rank(((PetscObject)part)->comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject)part)->comm, &size);CHKERRQ(ierr);
    nb_locals = mat->rmap->N / size;
    locals = parttab + rank * nb_locals;
    if (rank < mat->rmap->N % size) {
        nb_locals++;
        locals += rank;
    } else {
        locals += mat->rmap->N % size;
    }
    ierr = ISCreateGeneral(((PetscObject)part)->comm, nb_locals, locals, partitioning);CHKERRQ(ierr);

    /* destroying old objects */
    ierr = PetscFree(parttab);CHKERRQ(ierr);
    if (matSeq != mat) {
        ierr = MatDestroy(matSeq);CHKERRQ(ierr);
    }
    if (matMPI != mat) {
        ierr = MatDestroy(matMPI);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningView_Party"
PetscErrorCode MatPartitioningView_Party(MatPartitioning part, PetscViewer viewer)
{
  MatPartitioning_Party *party = (MatPartitioning_Party *) part->data;
  PetscErrorCode        ierr;
  PetscMPIInt           rank;
  PetscTruth            iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)part)->comm, &rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    if (!rank && party->mesg_log) {
      ierr = PetscViewerASCIIPrintf(viewer, "%s\n", party->mesg_log);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Viewer type %s not supported for this Party partitioner",((PetscObject) viewer)((PetscObject))->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetGlobal"
/*@C
     MatPartitioningPartySetGlobal - Set method for global partitioning.

  Input Parameter:
.  part - the partitioning context
.  method - May be one of MP_PARTY_OPT, MP_PARTY_LIN, MP_PARTY_SCA, 
    MP_PARTY_RAN, MP_PARTY_GBF, MP_PARTY_GCF, MP_PARTY_BUB or MP_PARTY_DEF, or
    alternatively a string describing the method. Two or more methods can be 
    combined like "gbf,gcf". Check the Party Library Users Manual for details.

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetGlobal(MatPartitioning part, const char *global)
{
    MatPartitioning_Party *party = (MatPartitioning_Party *) part->data;

    PetscFunctionBegin;
    PetscStrcpy(party->global_method, global);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetLocal"
/*@C
     MatPartitioningPartySetLocal - Set method for local partitioning.

  Input Parameter:
.  part - the partitioning context
.  method - One of MP_PARTY_HELPFUL_SETS, MP_PARTY_KERNIGHAN_LIN, or MP_PARTY_NONE. 
    Check the Party Library Users Manual for details.

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetLocal(MatPartitioning part, const char *local)
{
    MatPartitioning_Party *party = (MatPartitioning_Party *) part->data;

    PetscFunctionBegin;
    PetscStrcpy(party->local_method, local);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetCoarseLevel"
/*@
    MatPartitioningPartySetCoarseLevel - Set the coarse level 
    
  Input Parameter:
.  part - the partitioning context
.  level - the coarse level in range [0.0,1.0]

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetCoarseLevel(MatPartitioning part, PetscReal level)
{
    MatPartitioning_Party *party = (MatPartitioning_Party *) part->data;

    PetscFunctionBegin;
    if (level < 0 || level > 1.0) {
        SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Party: level of coarsening out of range [0.01-1.0]");
    } else {
        party->nbvtxcoarsed = (int)(part->adj->cmap->N * level);
    }
    if (party->nbvtxcoarsed < 20) party->nbvtxcoarsed = 20;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetMatchOptimization"
/*@
    MatPartitioningPartySetMatchOptimization - Activate matching optimization for graph reduction 
    
  Input Parameter:
.  part - the partitioning context
.  opt - activate optimization

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetMatchOptimization(MatPartitioning part,
    PetscTruth opt)
{
    MatPartitioning_Party *party = (MatPartitioning_Party *) part->data;

    PetscFunctionBegin;
    if (opt)
        PetscStrcpy(party->redo, "w3");
    else
        PetscStrcpy(party->redo, "");
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetBipart"
/*@
    MatPartitioningPartySetBipart - Activate or deactivate recursive bisection.
    
  Input Parameter:
.  part - the partitioning context
.  bp - PETSC_TRUE to activate recursive bisection 

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningPartySetBipart(MatPartitioning part, PetscTruth bp)
{
    MatPartitioning_Party *party = (MatPartitioning_Party *) part->data;

    PetscFunctionBegin;
    if (bp)
        party->rec = 1;
    else
        party->rec = 0;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningSetFromOptions_Party"
PetscErrorCode MatPartitioningSetFromOptions_Party(MatPartitioning part)
{
    PetscErrorCode ierr;
    PetscTruth flag, b;
    char value[15];
    PetscReal r;

    PetscFunctionBegin;
    ierr = PetscOptionsHead("Set Party partitioning options");CHKERRQ(ierr);

    ierr = PetscOptionsString("-mat_partitioning_party_global",
        "Global method to use", "MatPartitioningPartySetGlobal", "gcf,gbf",
        value, 15, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningPartySetGlobal(part, value);CHKERRQ(ierr);

    ierr = PetscOptionsString("-mat_partitioning_party_local",
        "Local method to use", "MatPartitioningPartySetLocal", "kl", value, 15,
        &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningPartySetLocal(part, value);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-mat_partitioning_party_coarse_level",
        "Coarse level", "MatPartitioningPartySetCoarseLevel", 0, &r,
        &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningPartySetCoarseLevel(part, r);CHKERRQ(ierr);

    ierr = PetscOptionsTruth("-mat_partitioning_party_match_optimization",
        "Matching optimization on/off (boolean)",
        "MatPartitioningPartySetMatchOptimization", PETSC_TRUE, &b, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningPartySetMatchOptimization(part, b);CHKERRQ(ierr);

    ierr = PetscOptionsTruth("-mat_partitioning_party_bipart",
        "Bipartitioning option on/off (boolean)",
        "MatPartitioningPartySetBipart", PETSC_TRUE, &b, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningPartySetBipart(part, b);CHKERRQ(ierr);

    ierr = PetscOptionsTail();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningDestroy_Party"
PetscErrorCode MatPartitioningDestroy_Party(MatPartitioning part)
{
    MatPartitioning_Party *party = (MatPartitioning_Party *) part->data;
    PetscErrorCode        ierr;

    PetscFunctionBegin;
    ierr = PetscFree(party->mesg_log);CHKERRQ(ierr);
    ierr = PetscFree(party);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*MC
   MAT_PARTITIONING_PARTY - Creates a partitioning context via the external package Party.

   Collective on MPI_Comm

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:
+  -mat_partitioning_party_global <gcf,gbf>: Global method to use (MatPartitioningPartySetGlobal)
.  -mat_partitioning_party_local <kl>: Local method to use (MatPartitioningPartySetLocal)
.  -mat_partitioning_party_coarse_level <0>: Coarse level (MatPartitioningPartySetCoarseLevel)
.  -mat_partitioning_party_match_optimization: <true> Matching optimization on/off (boolean) (MatPartitioningPartySetMatchOptimization)
-  -mat_partitioning_party_bipart: <true> Bipartitioning option on/off (boolean) (MatPartitioningPartySetBipart)

   Level: beginner

   Notes: See http://wwwcs.upb.de/fachbereich/AG/monien/RESEARCH/PART/party.html

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningCreate_Party"
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningCreate_Party(MatPartitioning part)
{
    PetscErrorCode ierr;
    MatPartitioning_Party *party;

    PetscFunctionBegin;
    ierr = PetscNewLog(part,MatPartitioning_Party, &party);CHKERRQ(ierr);
    part->data = (void*) party;

    PetscStrcpy(party->global_method, "gcf,gbf");
    PetscStrcpy(party->local_method, "kl");
    PetscStrcpy(party->redm, "lam");
    PetscStrcpy(party->redo, "w3");

    party->nbvtxcoarsed = 200;
    party->rec = 1;
    party->output = 1;
    party->mesg_log = NULL;

    part->ops->apply = MatPartitioningApply_Party;
    part->ops->view = MatPartitioningView_Party;
    part->ops->destroy = MatPartitioningDestroy_Party;
    part->ops->setfromoptions = MatPartitioningSetFromOptions_Party;
    PetscFunctionReturn(0);
}

EXTERN_C_END
