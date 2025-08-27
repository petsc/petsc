#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscdm.h>

/* linked list methods
 *
 *  PetscCDCreate
 */
PetscErrorCode PetscCDCreate(PetscInt a_size, PetscCoarsenData **a_out)
{
  PetscCoarsenData *ail;

  PetscFunctionBegin;
  /* allocate pool, partially */
  PetscCall(PetscNew(&ail));
  *a_out               = ail;
  ail->pool_list.next  = NULL;
  ail->pool_list.array = NULL;
  ail->chk_sz          = 0;
  /* allocate array */
  ail->size = a_size;
  PetscCall(PetscCalloc1(a_size, &ail->array));
  ail->extra_nodes = NULL;
  ail->mat         = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDDestroy
 */
PetscErrorCode PetscCDDestroy(PetscCoarsenData *ail)
{
  PetscCDArrNd *n = &ail->pool_list;

  PetscFunctionBegin;
  n = n->next;
  while (n) {
    PetscCDArrNd *lstn = n;

    n = n->next;
    PetscCall(PetscFree(lstn));
  }
  if (ail->pool_list.array) PetscCall(PetscFree(ail->pool_list.array));
  PetscCall(PetscFree(ail->array));
  if (ail->mat) PetscCall(MatDestroy(&ail->mat));
  /* delete this (+agg+pool array) */
  PetscCall(PetscFree(ail));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDSetChunkSize
 */
PetscErrorCode PetscCDSetChunkSize(PetscCoarsenData *ail, PetscInt a_sz)
{
  PetscFunctionBegin;
  ail->chk_sz = a_sz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*  PetscCDGetNewNode
 */
static PetscErrorCode PetscCDGetNewNode(PetscCoarsenData *ail, PetscCDIntNd **a_out, PetscInt a_id)
{
  PetscFunctionBegin;
  *a_out = NULL; /* squelch -Wmaybe-uninitialized */
  if (ail->extra_nodes) {
    PetscCDIntNd *node = ail->extra_nodes;

    ail->extra_nodes = node->next;
    node->gid        = a_id;
    node->next       = NULL;
    *a_out           = node;
  } else {
    if (!ail->pool_list.array) {
      if (!ail->chk_sz) ail->chk_sz = 10; /* use a chuck size of ail->size? */
      PetscCall(PetscMalloc1(ail->chk_sz, &ail->pool_list.array));
      ail->new_node       = ail->pool_list.array;
      ail->new_left       = ail->chk_sz;
      ail->new_node->next = NULL;
    } else if (!ail->new_left) {
      PetscCDArrNd *node;

      PetscCall(PetscMalloc(ail->chk_sz * sizeof(PetscCDIntNd) + sizeof(PetscCDArrNd), &node));
      node->array         = (PetscCDIntNd *)(node + 1);
      node->next          = ail->pool_list.next;
      ail->pool_list.next = node;
      ail->new_left       = ail->chk_sz;
      ail->new_node       = node->array;
    }
    ail->new_node->gid  = a_id;
    ail->new_node->next = NULL;
    *a_out              = ail->new_node++;
    ail->new_left--;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDIntNdSetID
 */
PetscErrorCode PetscCDIntNdSetID(PetscCDIntNd *a_this, PetscInt a_id)
{
  PetscFunctionBegin;
  a_this->gid = a_id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDIntNdGetID
 */
PetscErrorCode PetscCDIntNdGetID(const PetscCDIntNd *a_this, PetscInt *a_gid)
{
  PetscFunctionBegin;
  *a_gid = a_this->gid;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDGetHeadPos
 */
PetscErrorCode PetscCDGetHeadPos(const PetscCoarsenData *ail, PetscInt a_idx, PetscCDIntNd **pos)
{
  PetscFunctionBegin;
  PetscCheck(a_idx < ail->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "a_idx >= ail->size: a_idx=%" PetscInt_FMT ".", a_idx);
  *pos = ail->array[a_idx];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDGetNextPos
 */
PetscErrorCode PetscCDGetNextPos(const PetscCoarsenData *ail, PetscInt l_idx, PetscCDIntNd **pos)
{
  PetscFunctionBegin;
  PetscCheck(*pos, PETSC_COMM_SELF, PETSC_ERR_PLIB, "NULL input position.");
  *pos = (*pos)->next;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDAppendID
 */
PetscErrorCode PetscCDAppendID(PetscCoarsenData *ail, PetscInt a_idx, PetscInt a_id)
{
  PetscCDIntNd *n, *n2;

  PetscFunctionBegin;
  PetscCall(PetscCDGetNewNode(ail, &n, a_id));
  PetscCheck(a_idx < ail->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %" PetscInt_FMT " out of range.", a_idx);
  if (!(n2 = ail->array[a_idx])) ail->array[a_idx] = n;
  else {
    do {
      if (!n2->next) {
        n2->next = n;
        PetscCheck(!n->next, PETSC_COMM_SELF, PETSC_ERR_PLIB, "n should not have a next");
        break;
      }
      n2 = n2->next;
    } while (n2);
    PetscCheck(n2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "n2 should be non-null");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDAppendNode
 */
PetscErrorCode PetscCDAppendNode(PetscCoarsenData *ail, PetscInt a_idx, PetscCDIntNd *a_n)
{
  PetscCDIntNd *n2;

  PetscFunctionBegin;
  PetscCheck(a_idx < ail->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %" PetscInt_FMT " out of range.", a_idx);
  if (!(n2 = ail->array[a_idx])) ail->array[a_idx] = a_n;
  else {
    do {
      if (!n2->next) {
        n2->next  = a_n;
        a_n->next = NULL;
        break;
      }
      n2 = n2->next;
    } while (n2);
    PetscCheck(n2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "n2 should be non-null");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDRemoveNextNode: a_last->next, this exposes single linked list structure to API (not used)
 */
PetscErrorCode PetscCDRemoveNextNode(PetscCoarsenData *ail, PetscInt a_idx, PetscCDIntNd *a_last)
{
  PetscCDIntNd *del;

  PetscFunctionBegin;
  PetscCheck(a_idx < ail->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %" PetscInt_FMT " out of range.", a_idx);
  PetscCheck(a_last->next, PETSC_COMM_SELF, PETSC_ERR_PLIB, "a_last should have a next");
  del          = a_last->next;
  a_last->next = del->next;
  /* del->next = NULL; -- this still used in a iterator so keep it intact -- need to fix this with a double linked list */
  /* could reuse n2 but PetscCDAppendNode sometimes uses it */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDPrint
 */
PetscErrorCode PetscCDPrint(const PetscCoarsenData *ail, PetscInt Istart, MPI_Comm comm)
{
  PetscCDIntNd *n, *n2;
  PetscInt      ii;

  PetscFunctionBegin;
  for (ii = 0; ii < ail->size; ii++) {
    n2 = n = ail->array[ii];
    if (n) PetscCall(PetscSynchronizedPrintf(comm, "list %" PetscInt_FMT ":", ii + Istart));
    while (n) {
      PetscCall(PetscSynchronizedPrintf(comm, " %" PetscInt_FMT, n->gid));
      n = n->next;
    }
    if (n2) PetscCall(PetscSynchronizedPrintf(comm, "\n"));
  }
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDMoveAppend - take list in a_srcidx and appends to destidx
 */
PetscErrorCode PetscCDMoveAppend(PetscCoarsenData *ail, PetscInt a_destidx, PetscInt a_srcidx)
{
  PetscCDIntNd *n;

  PetscFunctionBegin;
  PetscCheck(a_srcidx < ail->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %" PetscInt_FMT " out of range.", a_srcidx);
  PetscCheck(a_destidx < ail->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %" PetscInt_FMT " out of range.", a_destidx);
  PetscCheck(a_destidx != a_srcidx, PETSC_COMM_SELF, PETSC_ERR_PLIB, "a_destidx==a_srcidx %" PetscInt_FMT ".", a_destidx);
  n = ail->array[a_destidx];
  if (!n) ail->array[a_destidx] = ail->array[a_srcidx];
  else {
    do {
      if (!n->next) {
        n->next = ail->array[a_srcidx]; // append
        break;
      }
      n = n->next;
    } while (1);
  }
  ail->array[a_srcidx] = NULL; // empty
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDRemoveAllAt - empty one list and move data to cache
 */
PetscErrorCode PetscCDRemoveAllAt(PetscCoarsenData *ail, PetscInt a_idx)
{
  PetscCDIntNd *rem, *n1;

  PetscFunctionBegin;
  PetscCheck(a_idx < ail->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %" PetscInt_FMT " out of range.", a_idx);
  rem               = ail->array[a_idx];
  ail->array[a_idx] = NULL;
  if (!(n1 = ail->extra_nodes)) ail->extra_nodes = rem;
  else {
    while (n1->next) n1 = n1->next;
    n1->next = rem;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDCountAt
 */
PetscErrorCode PetscCDCountAt(const PetscCoarsenData *ail, PetscInt a_idx, PetscInt *a_sz)
{
  PetscCDIntNd *n1;
  PetscInt      sz = 0;

  PetscFunctionBegin;
  PetscCheck(a_idx < ail->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %" PetscInt_FMT " out of range.", a_idx);
  n1 = ail->array[a_idx];
  while (n1) {
    n1 = n1->next;
    sz++;
  }
  *a_sz = sz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDSize
 */
PetscErrorCode PetscCDCount(const PetscCoarsenData *ail, PetscInt *a_sz)
{
  PetscInt sz = 0;

  PetscFunctionBegin;
  for (PetscInt ii = 0; ii < ail->size; ii++) {
    PetscCDIntNd *n1 = ail->array[ii];

    while (n1) {
      n1 = n1->next;
      sz++;
    }
  }
  *a_sz = sz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDIsEmptyAt - Is the list empty? (not used)
 */
PetscErrorCode PetscCDIsEmptyAt(const PetscCoarsenData *ail, PetscInt a_idx, PetscBool *a_e)
{
  PetscFunctionBegin;
  PetscCheck(a_idx < ail->size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %" PetscInt_FMT " out of range.", a_idx);
  *a_e = (PetscBool)(ail->array[a_idx] == NULL);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDGetNonemptyIS - used for C-F methods
 */
PetscErrorCode PetscCDGetNonemptyIS(PetscCoarsenData *ail, IS *a_mis)
{
  PetscCDIntNd *n;
  PetscInt      ii, kk;
  PetscInt     *permute;

  PetscFunctionBegin;
  for (ii = kk = 0; ii < ail->size; ii++) {
    n = ail->array[ii];
    if (n) kk++;
  }
  PetscCall(PetscMalloc1(kk, &permute));
  for (ii = kk = 0; ii < ail->size; ii++) {
    n = ail->array[ii];
    if (n) permute[kk++] = ii;
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, kk, permute, PETSC_OWN_POINTER, a_mis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDGetMat
 */
PetscErrorCode PetscCDGetMat(PetscCoarsenData *ail, Mat *a_mat)
{
  PetscFunctionBegin;
  *a_mat = ail->mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDSetMat
 */
PetscErrorCode PetscCDSetMat(PetscCoarsenData *ail, Mat a_mat)
{
  PetscFunctionBegin;
  if (ail->mat) PetscCall(MatDestroy(&ail->mat)); //should not happen
  ail->mat = a_mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDClearMat
 */
PetscErrorCode PetscCDClearMat(PetscCoarsenData *ail)
{
  PetscFunctionBegin;
  ail->mat = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscCDGetASMBlocks - get IS of aggregates for ASM smoothers
 */
PetscErrorCode PetscCDGetASMBlocks(const PetscCoarsenData *ail, const PetscInt a_bs, PetscInt *a_sz, IS **a_local_is)
{
  PetscCDIntNd *n;
  PetscInt      lsz, ii, kk, *idxs, jj, gid;
  IS           *is_loc = NULL;

  PetscFunctionBegin;
  for (ii = kk = 0; ii < ail->size; ii++) {
    if (ail->array[ii]) kk++;
  }
  *a_sz = kk;
  PetscCall(PetscMalloc1(kk, &is_loc));
  for (ii = kk = 0; ii < ail->size; ii++) {
    for (lsz = 0, n = ail->array[ii]; n; lsz++, n = n->next) /* void */
      ;
    if (lsz) {
      PetscCall(PetscMalloc1(a_bs * lsz, &idxs));
      for (lsz = 0, n = ail->array[ii]; n; n = n->next) {
        PetscCall(PetscCDIntNdGetID(n, &gid));
        for (jj = 0; jj < a_bs; lsz++, jj++) idxs[lsz] = a_bs * gid + jj;
      }
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, lsz, idxs, PETSC_OWN_POINTER, &is_loc[kk++]));
    }
  }
  PetscCheck(*a_sz == kk, PETSC_COMM_SELF, PETSC_ERR_PLIB, "*a_sz %" PetscInt_FMT " != kk %" PetscInt_FMT, *a_sz, kk);
  *a_local_is = is_loc; /* out */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* edge for priority queue */
typedef struct edge_tag {
  PetscReal weight;
  PetscInt  lid0, gid1, ghost1_idx;
} Edge;

#define MY_MEPS (PETSC_MACHINE_EPSILON * 100)
static int gamg_hem_compare(const void *a, const void *b)
{
  PetscReal va = ((Edge *)a)->weight, vb = ((Edge *)b)->weight;
  return (va <= vb - MY_MEPS) ? 1 : (va > vb + MY_MEPS) ? -1 : 0; /* 0 for equal */
}

/*
  MatCoarsenApply_HEM_private - parallel heavy edge matching

  Input Parameter:
   . a_Gmat - global matrix of the graph
   . n_iter - number of matching iterations
   . threshold - threshold for filtering graphs

  Output Parameter:
   . a_locals_llist - array of list of local nodes rooted at local node
*/
static PetscErrorCode MatCoarsenApply_HEM_private(Mat a_Gmat, const PetscInt n_iter, const PetscReal threshold, PetscCoarsenData **a_locals_llist)
{
#define REQ_BF_SIZE 100
  PetscBool         isMPI;
  MPI_Comm          comm;
  PetscInt          ix, *ii, *aj, Istart, bc_agg = -1, *rbuff = NULL, rbuff_sz = 0;
  PetscMPIInt       rank, size, comm_procs[REQ_BF_SIZE], ncomm_procs, *lid_max_pe;
  const PetscInt    nloc = a_Gmat->rmap->n, request_size = PetscCeilInt((int)sizeof(MPI_Request), (int)sizeof(PetscInt));
  PetscInt         *lid_cprowID;
  PetscBool        *lid_matched;
  Mat_SeqAIJ       *matA, *matB = NULL;
  Mat_MPIAIJ       *mpimat     = NULL;
  PetscScalar       one        = 1.;
  PetscCoarsenData *agg_llists = NULL, *ghost_deleted_list = NULL, *bc_list = NULL;
  Mat               cMat, tMat, P;
  MatScalar        *ap;
  IS                info_is;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)a_Gmat, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatGetOwnershipRange(a_Gmat, &Istart, NULL));
  PetscCall(ISCreate(comm, &info_is));
  PetscCall(PetscInfo(info_is, "Start %" PetscInt_FMT " iterations of HEM.\n", n_iter));

  PetscCall(PetscMalloc3(nloc, &lid_matched, nloc, &lid_cprowID, nloc, &lid_max_pe));
  PetscCall(PetscCDCreate(nloc, &agg_llists));
  PetscCall(PetscCDSetChunkSize(agg_llists, nloc + 1));
  *a_locals_llist = agg_llists;
  /* add self to all lists */
  for (PetscInt kk = 0; kk < nloc; kk++) PetscCall(PetscCDAppendID(agg_llists, kk, Istart + kk));
  /* make a copy of the graph, this gets destroyed in iterates */
  PetscCall(MatDuplicate(a_Gmat, MAT_COPY_VALUES, &cMat));
  PetscCall(MatConvert(cMat, MATAIJ, MAT_INPLACE_MATRIX, &cMat));
  isMPI = (PetscBool)(size > 1);
  if (isMPI) {
    /* list of deleted ghosts, should compress this */
    PetscCall(PetscCDCreate(size, &ghost_deleted_list));
    PetscCall(PetscCDSetChunkSize(ghost_deleted_list, 100));
  }
  for (PetscInt iter = 0; iter < n_iter; iter++) {
    const PetscScalar *lghost_max_ew, *lid_max_ew;
    PetscBool         *lghost_matched;
    PetscMPIInt       *lghost_pe, *lghost_max_pe;
    Vec                locMaxEdge, ghostMaxEdge, ghostMaxPE, locMaxPE;
    PetscInt          *lghost_gid, nEdges, nEdges0, num_ghosts = 0;
    Edge              *Edges;
    const PetscInt     n_sub_its = 1000; // in case of a bug, stop at some point

    /* get submatrices of cMat */
    for (PetscInt kk = 0; kk < nloc; kk++) lid_cprowID[kk] = -1;
    if (isMPI) {
      mpimat = (Mat_MPIAIJ *)cMat->data;
      matA   = (Mat_SeqAIJ *)mpimat->A->data;
      matB   = (Mat_SeqAIJ *)mpimat->B->data;
      if (!matB->compressedrow.use) {
        /* force construction of compressed row data structure since code below requires it */
        PetscCall(MatCheckCompressedRow(mpimat->B, matB->nonzerorowcnt, &matB->compressedrow, matB->i, mpimat->B->rmap->n, -1.0));
      }
      /* set index into compressed row 'lid_cprowID' */
      for (ix = 0; ix < matB->compressedrow.nrows; ix++) {
        PetscInt *ridx = matB->compressedrow.rindex, lid = ridx[ix];
        if (ridx[ix] >= 0) lid_cprowID[lid] = ix;
      }
    } else {
      matA = (Mat_SeqAIJ *)cMat->data;
    }
    /* set matched flags: true for empty list */
    for (PetscInt kk = 0; kk < nloc; kk++) {
      PetscCall(PetscCDCountAt(agg_llists, kk, &ix));
      if (ix > 0) lid_matched[kk] = PETSC_FALSE;
      else lid_matched[kk] = PETSC_TRUE; // call deleted gids as matched
    }
    /* max edge and pe vecs */
    PetscCall(MatCreateVecs(cMat, &locMaxEdge, NULL));
    PetscCall(MatCreateVecs(cMat, &locMaxPE, NULL));
    /* get 'lghost_pe' & 'lghost_gid' & init. 'lghost_matched' using 'mpimat->lvec' */
    if (isMPI) {
      Vec                vec;
      PetscScalar        vval;
      const PetscScalar *buf;

      PetscCall(MatCreateVecs(cMat, &vec, NULL));
      PetscCall(VecGetLocalSize(mpimat->lvec, &num_ghosts));
      /* lghost_matched */
      for (PetscInt kk = 0, gid = Istart; kk < nloc; kk++, gid++) {
        PetscScalar vval = lid_matched[kk] ? 1.0 : 0.0;

        PetscCall(VecSetValues(vec, 1, &gid, &vval, INSERT_VALUES));
      }
      PetscCall(VecAssemblyBegin(vec));
      PetscCall(VecAssemblyEnd(vec));
      PetscCall(VecScatterBegin(mpimat->Mvctx, vec, mpimat->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mpimat->Mvctx, vec, mpimat->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetArrayRead(mpimat->lvec, &buf)); /* get proc ID in 'buf' */
      PetscCall(PetscMalloc4(num_ghosts, &lghost_matched, num_ghosts, &lghost_pe, num_ghosts, &lghost_gid, num_ghosts, &lghost_max_pe));

      for (PetscInt kk = 0; kk < num_ghosts; kk++) lghost_matched[kk] = (PetscBool)(PetscRealPart(buf[kk]) != 0); // the proc of the ghost for now
      PetscCall(VecRestoreArrayRead(mpimat->lvec, &buf));
      /* lghost_pe */
      vval = (PetscScalar)rank;
      for (PetscInt kk = 0, gid = Istart; kk < nloc; kk++, gid++) PetscCall(VecSetValues(vec, 1, &gid, &vval, INSERT_VALUES)); /* set with GID */
      PetscCall(VecAssemblyBegin(vec));
      PetscCall(VecAssemblyEnd(vec));
      PetscCall(VecScatterBegin(mpimat->Mvctx, vec, mpimat->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mpimat->Mvctx, vec, mpimat->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetArrayRead(mpimat->lvec, &buf));                                                   /* get proc ID in 'buf' */
      for (PetscInt kk = 0; kk < num_ghosts; kk++) lghost_pe[kk] = (PetscMPIInt)PetscRealPart(buf[kk]); // the proc of the ghost for now
      PetscCall(VecRestoreArrayRead(mpimat->lvec, &buf));
      /* lghost_gid */
      for (PetscInt kk = 0, gid = Istart; kk < nloc; kk++, gid++) {
        vval = (PetscScalar)gid;

        PetscCall(VecSetValues(vec, 1, &gid, &vval, INSERT_VALUES)); /* set with GID */
      }
      PetscCall(VecAssemblyBegin(vec));
      PetscCall(VecAssemblyEnd(vec));
      PetscCall(VecScatterBegin(mpimat->Mvctx, vec, mpimat->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mpimat->Mvctx, vec, mpimat->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecDestroy(&vec));
      PetscCall(VecGetArrayRead(mpimat->lvec, &buf)); /* get proc ID in 'lghost_gid' */
      for (PetscInt kk = 0; kk < num_ghosts; kk++) lghost_gid[kk] = (PetscInt)PetscRealPart(buf[kk]);
      PetscCall(VecRestoreArrayRead(mpimat->lvec, &buf));
    }
    // get 'comm_procs' (could hoist)
    for (PetscInt kk = 0; kk < REQ_BF_SIZE; kk++) comm_procs[kk] = -1;
    for (ix = 0, ncomm_procs = 0; ix < num_ghosts; ix++) {
      PetscMPIInt proc = lghost_pe[ix], idx = -1;

      for (PetscMPIInt k = 0; k < ncomm_procs && idx == -1; k++)
        if (comm_procs[k] == proc) idx = k;
      if (idx == -1) comm_procs[ncomm_procs++] = proc;
      PetscCheck(ncomm_procs != REQ_BF_SIZE, PETSC_COMM_SELF, PETSC_ERR_SUP, "Receive request array too small: %d", ncomm_procs);
    }
    /* count edges, compute initial 'locMaxEdge', 'locMaxPE' */
    nEdges0 = 0;
    for (PetscInt kk = 0, gid = Istart; kk < nloc; kk++, gid++) {
      PetscReal   max_e = 0., tt;
      PetscScalar vval;
      PetscInt    lid = kk, max_pe = rank, pe, n;

      ii = matA->i;
      n  = ii[lid + 1] - ii[lid];
      aj = PetscSafePointerPlusOffset(matA->j, ii[lid]);
      ap = PetscSafePointerPlusOffset(matA->a, ii[lid]);
      for (PetscInt jj = 0; jj < n; jj++) {
        PetscInt lidj = aj[jj];

        if ((tt = PetscRealPart(ap[jj])) > threshold && lidj != lid) {
          if (tt > max_e) max_e = tt;
          if (lidj > lid) nEdges0++;
        }
      }
      if ((ix = lid_cprowID[lid]) != -1) { /* if I have any ghost neighbors */
        ii = matB->compressedrow.i;
        n  = ii[ix + 1] - ii[ix];
        ap = matB->a + ii[ix];
        aj = matB->j + ii[ix];
        for (PetscInt jj = 0; jj < n; jj++) {
          if ((tt = PetscRealPart(ap[jj])) > threshold) {
            if (tt > max_e) max_e = tt;
            nEdges0++;
            if ((pe = lghost_pe[aj[jj]]) > max_pe) max_pe = pe;
          }
        }
      }
      vval = max_e;
      PetscCall(VecSetValues(locMaxEdge, 1, &gid, &vval, INSERT_VALUES));
      vval = (PetscScalar)max_pe;
      PetscCall(VecSetValues(locMaxPE, 1, &gid, &vval, INSERT_VALUES));
      if (iter == 0 && max_e <= MY_MEPS) { // add BCs to fake aggregate
        lid_matched[lid] = PETSC_TRUE;
        if (bc_agg == -1) {
          bc_agg = lid;
          PetscCall(PetscCDCreate(1, &bc_list));
        }
        PetscCall(PetscCDRemoveAllAt(agg_llists, lid));
        PetscCall(PetscCDAppendID(bc_list, 0, Istart + lid));
      }
    }
    PetscCall(VecAssemblyBegin(locMaxEdge));
    PetscCall(VecAssemblyEnd(locMaxEdge));
    PetscCall(VecAssemblyBegin(locMaxPE));
    PetscCall(VecAssemblyEnd(locMaxPE));
    /* make 'ghostMaxEdge_max_ew', 'lghost_max_pe' */
    if (isMPI) {
      const PetscScalar *buf;

      PetscCall(VecDuplicate(mpimat->lvec, &ghostMaxEdge));
      PetscCall(VecScatterBegin(mpimat->Mvctx, locMaxEdge, ghostMaxEdge, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mpimat->Mvctx, locMaxEdge, ghostMaxEdge, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecDuplicate(mpimat->lvec, &ghostMaxPE));
      PetscCall(VecScatterBegin(mpimat->Mvctx, locMaxPE, ghostMaxPE, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mpimat->Mvctx, locMaxPE, ghostMaxPE, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetArrayRead(ghostMaxPE, &buf));
      for (PetscInt kk = 0; kk < num_ghosts; kk++) lghost_max_pe[kk] = (PetscMPIInt)PetscRealPart(buf[kk]); // the MAX proc of the ghost now
      PetscCall(VecRestoreArrayRead(ghostMaxPE, &buf));
    }
    { // make lid_max_pe
      const PetscScalar *buf;

      PetscCall(VecGetArrayRead(locMaxPE, &buf));
      for (PetscInt kk = 0; kk < nloc; kk++) lid_max_pe[kk] = (PetscMPIInt)PetscRealPart(buf[kk]); // the MAX proc of the ghost now
      PetscCall(VecRestoreArrayRead(locMaxPE, &buf));
    }
    /* setup sorted list of edges, and make 'Edges' */
    PetscCall(PetscMalloc1(nEdges0, &Edges));
    nEdges = 0;
    for (PetscInt kk = 0, n; kk < nloc; kk++) {
      const PetscInt lid = kk;
      PetscReal      tt;

      ii = matA->i;
      n  = ii[lid + 1] - ii[lid];
      aj = PetscSafePointerPlusOffset(matA->j, ii[lid]);
      ap = PetscSafePointerPlusOffset(matA->a, ii[lid]);
      for (PetscInt jj = 0; jj < n; jj++) {
        PetscInt lidj = aj[jj];

        if ((tt = PetscRealPart(ap[jj])) > threshold && lidj != lid) {
          if (lidj > lid) {
            Edges[nEdges].lid0       = lid;
            Edges[nEdges].gid1       = lidj + Istart;
            Edges[nEdges].ghost1_idx = -1;
            Edges[nEdges].weight     = tt;
            nEdges++;
          }
        }
      }
      if ((ix = lid_cprowID[lid]) != -1) { /* if I have any ghost neighbor */
        ii = matB->compressedrow.i;
        n  = ii[ix + 1] - ii[ix];
        ap = matB->a + ii[ix];
        aj = matB->j + ii[ix];
        for (PetscInt jj = 0; jj < n; jj++) {
          if ((tt = PetscRealPart(ap[jj])) > threshold) {
            Edges[nEdges].lid0       = lid;
            Edges[nEdges].gid1       = lghost_gid[aj[jj]];
            Edges[nEdges].ghost1_idx = aj[jj];
            Edges[nEdges].weight     = tt;
            nEdges++;
          }
        }
      }
    }
    PetscCheck(nEdges == nEdges0, PETSC_COMM_SELF, PETSC_ERR_SUP, "nEdges != nEdges0: %" PetscInt_FMT " %" PetscInt_FMT, nEdges0, nEdges);
    if (Edges) qsort(Edges, nEdges, sizeof(Edge), gamg_hem_compare);

    PetscCall(PetscInfo(info_is, "[%d] HEM iteration %" PetscInt_FMT " with %" PetscInt_FMT " edges\n", rank, iter, nEdges));

    /* projection matrix */
    PetscCall(MatCreate(comm, &P));
    PetscCall(MatSetType(P, MATAIJ));
    PetscCall(MatSetSizes(P, nloc, nloc, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatMPIAIJSetPreallocation(P, 1, NULL, 1, NULL));
    PetscCall(MatSeqAIJSetPreallocation(P, 1, NULL));
    PetscCall(MatSetUp(P));
    /* process - communicate - process */
    for (PetscInt sub_it = 0, old_num_edge = 0; /* sub_it < n_sub_its */; /* sub_it++ */) {
      PetscInt    nactive_edges = 0, n_act_n[3], gn_act_n[3];
      PetscMPIInt tag1, tag2;

      PetscCall(VecGetArrayRead(locMaxEdge, &lid_max_ew));
      if (isMPI) {
        PetscCall(VecGetArrayRead(ghostMaxEdge, &lghost_max_ew));
        PetscCall(PetscCommGetNewTag(comm, &tag1));
        PetscCall(PetscCommGetNewTag(comm, &tag2));
      }
      for (PetscInt kk = 0; kk < nEdges; kk++) {
        const Edge    *e    = &Edges[kk];
        const PetscInt lid0 = e->lid0, gid1 = e->gid1, ghost1_idx = e->ghost1_idx, gid0 = lid0 + Istart, lid1 = gid1 - Istart;
        PetscBool      isOK = PETSC_TRUE, print = PETSC_FALSE;

        if (print)
          PetscCall(PetscSynchronizedPrintf(comm, "\t[%d] edge (%" PetscInt_FMT " %" PetscInt_FMT "), %s %s %s\n", rank, gid0, gid1, lid_matched[lid0] ? "true" : "false", (ghost1_idx != -1 && lghost_matched[ghost1_idx]) ? "true" : "false", (ghost1_idx == -1 && lid_matched[lid1]) ? "true" : "false"));
        /* skip if either vertex is matched already */
        if (lid_matched[lid0] || (ghost1_idx != -1 && lghost_matched[ghost1_idx]) || (ghost1_idx == -1 && lid_matched[lid1])) continue;

        nactive_edges++;
        PetscCheck(PetscRealPart(lid_max_ew[lid0]) >= e->weight - MY_MEPS, PETSC_COMM_SELF, PETSC_ERR_SUP, "edge weight %e > max %e", (double)e->weight, (double)PetscRealPart(lid_max_ew[lid0]));
        if (print) PetscCall(PetscSynchronizedPrintf(comm, "\t[%d] active edge (%" PetscInt_FMT " %" PetscInt_FMT "), diff0 = %10.4e\n", rank, gid0, gid1, (double)(PetscRealPart(lid_max_ew[lid0]) - e->weight)));
        // smaller edge, lid_max_ew get updated - e0
        if (PetscRealPart(lid_max_ew[lid0]) > e->weight + MY_MEPS) {
          if (print)
            PetscCall(PetscSynchronizedPrintf(comm, "\t\t[%d] 1) e0 SKIPPING small edge %20.14e edge (%" PetscInt_FMT " %" PetscInt_FMT "), diff = %10.4e to proc %d. max = %20.14e, w = %20.14e\n", rank, (double)e->weight, gid0, gid1, (double)(PetscRealPart(lid_max_ew[lid0]) - e->weight), ghost1_idx != -1 ? lghost_pe[ghost1_idx] : rank, (double)PetscRealPart(lid_max_ew[lid0]),
                                              (double)e->weight));
          continue; // we are basically filter edges here
        }
        // e1 - local
        if (ghost1_idx == -1) {
          if (PetscRealPart(lid_max_ew[lid1]) > e->weight + MY_MEPS) {
            if (print)
              PetscCall(PetscSynchronizedPrintf(comm, "\t\t%c[%d] 2) e1 SKIPPING small local edge %20.14e edge (%" PetscInt_FMT " %" PetscInt_FMT "), diff = %10.4e\n", ghost1_idx != -1 ? '\t' : ' ', rank, (double)e->weight, gid0, gid1, (double)(PetscRealPart(lid_max_ew[lid1]) - e->weight)));
            continue; // we are basically filter edges here
          }
        } else { // e1 - ghost
          /* see if edge might get matched on other proc */
          PetscReal g_max_e1 = PetscRealPart(lghost_max_ew[ghost1_idx]);

          if (print)
            PetscCall(PetscSynchronizedPrintf(comm, "\t\t\t[%d] CHECK GHOST e1, edge (%" PetscInt_FMT " %" PetscInt_FMT "), E0 MAX EDGE WEIGHT = %10.4e, EDGE WEIGHT = %10.4e, diff1 = %10.4e, ghost proc %d with max pe %d on e0 and %d on e1\n", rank, gid0, gid1, (double)PetscRealPart(lid_max_ew[lid0]),
                                              (double)e->weight, (double)(PetscRealPart(lghost_max_ew[ghost1_idx]) - e->weight), lghost_pe[ghost1_idx], lid_max_pe[lid0], lghost_max_pe[ghost1_idx]));
          if (g_max_e1 > e->weight + MY_MEPS) {
            /* PetscCall(PetscSynchronizedPrintf(comm,"\t\t\t\t[%d] 3) ghost e1 SKIPPING small edge (%d %d), diff = %10.4e from proc %d with max pe %d. max = %20.14e, w = %20.14e\n", rank, gid0, gid1, g_max_e1 - e->weight, lghost_pe[ghost1_idx], lghost_max_pe[ghost1_idx], g_max_e1, e->weight )); */
            continue;
          } else if (g_max_e1 >= e->weight - MY_MEPS && lghost_pe[ghost1_idx] > rank) { // is 'lghost_max_pe[ghost1_idx] > rank' needed?
            /* check for max_ea == to this edge and larger processor that will deal with this */
            if (print)
              PetscCall(PetscSynchronizedPrintf(comm, "\t\t\t[%d] ghost e1 SKIPPING EQUAL (%" PetscInt_FMT " %" PetscInt_FMT "), diff = %10.4e from larger proc %d with max pe %d. max = %20.14e, w = %20.14e\n", rank, gid0, gid1, (double)(PetscRealPart(lid_max_ew[lid0]) - e->weight), lghost_pe[ghost1_idx], lghost_max_pe[ghost1_idx], (double)g_max_e1,
                                                (double)e->weight));
            continue;
          } else {
            /* PetscCall(PetscSynchronizedPrintf(comm,"\t[%d] Edge (%d %d) passes gid0 tests, diff = %10.4e from proc %d with max pe %d. max = %20.14e, w = %20.14e\n", rank, gid0, gid1, g_max_e1 - e->weight, lghost_pe[ghost1_idx], lghost_max_pe[ghost1_idx], g_max_e1, e->weight )); */
          }
        }
        /* check ghost for v0 */
        if (isOK) {
          PetscReal max_e, ew;

          if ((ix = lid_cprowID[lid0]) != -1) { /* if I have any ghost neighbors */
            PetscInt n;

            ii = matB->compressedrow.i;
            n  = ii[ix + 1] - ii[ix];
            ap = matB->a + ii[ix];
            aj = matB->j + ii[ix];
            for (PetscInt jj = 0; jj < n && isOK; jj++) {
              PetscInt lidj = aj[jj];

              if (lghost_matched[lidj]) continue;
              ew = PetscRealPart(ap[jj]);
              if (ew <= threshold) continue;
              max_e = PetscRealPart(lghost_max_ew[lidj]);

              /* check for max_e == to this edge and larger processor that will deal with this */
              if (ew >= PetscRealPart(lid_max_ew[lid0]) - MY_MEPS && lghost_max_pe[lidj] > rank) isOK = PETSC_FALSE;
              PetscCheck(ew <= max_e + MY_MEPS, PETSC_COMM_SELF, PETSC_ERR_SUP, "edge weight %e > max %e. ncols = %" PetscInt_FMT ", gid0 = %" PetscInt_FMT ", gid1 = %" PetscInt_FMT, (double)PetscRealPart(ew), (double)PetscRealPart(max_e), n, lid0 + Istart, lghost_gid[lidj]);
              if (print)
                PetscCall(PetscSynchronizedPrintf(comm, "\t\t\t\t[%d] e0: looked at ghost adj (%" PetscInt_FMT " %" PetscInt_FMT "), diff = %10.4e, ghost on proc %d (max %d). isOK = %d, %d %d %d; ew = %e, lid0 max ew = %e, diff = %e, eps = %e\n", rank, gid0, lghost_gid[lidj], (double)(max_e - ew), lghost_pe[lidj], lghost_max_pe[lidj], isOK, (double)ew >= (double)(max_e - MY_MEPS), ew >= PetscRealPart(lid_max_ew[lid0]) - MY_MEPS, lghost_pe[lidj] > rank, (double)ew, (double)PetscRealPart(lid_max_ew[lid0]), (double)(ew - PetscRealPart(lid_max_ew[lid0])), (double)MY_MEPS));
            }
            if (!isOK && print) PetscCall(PetscSynchronizedPrintf(comm, "\t\t[%d] skip edge (%" PetscInt_FMT " %" PetscInt_FMT ") from ghost inspection\n", rank, gid0, gid1));
          }
          /* check local v1 */
          if (ghost1_idx == -1) {
            if ((ix = lid_cprowID[lid1]) != -1) { /* if I have any ghost neighbors */
              PetscInt n;

              ii = matB->compressedrow.i;
              n  = ii[ix + 1] - ii[ix];
              ap = matB->a + ii[ix];
              aj = matB->j + ii[ix];
              for (PetscInt jj = 0; jj < n && isOK; jj++) {
                PetscInt lidj = aj[jj];

                if (lghost_matched[lidj]) continue;
                ew = PetscRealPart(ap[jj]);
                if (ew <= threshold) continue;
                max_e = PetscRealPart(lghost_max_ew[lidj]);
                /* check for max_e == to this edge and larger processor that will deal with this */
                if (ew >= PetscRealPart(lid_max_ew[lid1]) - MY_MEPS && lghost_max_pe[lidj] > rank) isOK = PETSC_FALSE;
                PetscCheck(ew <= max_e + MY_MEPS, PETSC_COMM_SELF, PETSC_ERR_SUP, "edge weight %e > max %e", (double)PetscRealPart(ew), (double)PetscRealPart(max_e));
                if (print)
                  PetscCall(PetscSynchronizedPrintf(comm, "\t\t\t\t\t[%d] e1: looked at ghost adj (%" PetscInt_FMT " %" PetscInt_FMT "), diff = %10.4e, ghost on proc %d (max %d)\n", rank, gid0, lghost_gid[lidj], (double)(max_e - ew), lghost_pe[lidj], lghost_max_pe[lidj]));
              }
            }
            if (!isOK && print) PetscCall(PetscSynchronizedPrintf(comm, "\t\t[%d] skip edge (%" PetscInt_FMT " %" PetscInt_FMT ") from ghost inspection\n", rank, gid0, gid1));
          }
        }
        PetscReal e1_max_w = (ghost1_idx == -1 ? PetscRealPart(lid_max_ew[lid0]) : PetscRealPart(lghost_max_ew[ghost1_idx]));
        if (print)
          PetscCall(PetscSynchronizedPrintf(comm, "\t[%d] MATCHING (%" PetscInt_FMT " %" PetscInt_FMT ") e1 max weight = %e, e1 weight diff %e, %s. isOK = %d\n", rank, gid0, gid1, (double)e1_max_w, (double)(e1_max_w - e->weight), ghost1_idx == -1 ? "local" : "ghost", isOK));
        /* do it */
        if (isOK) {
          if (ghost1_idx == -1) {
            PetscCheck(!lid_matched[lid1], PETSC_COMM_SELF, PETSC_ERR_SUP, "local %" PetscInt_FMT " is matched", gid1);
            lid_matched[lid1] = PETSC_TRUE;                       /* keep track of what we've done this round */
            PetscCall(PetscCDMoveAppend(agg_llists, lid0, lid1)); // takes lid1's list and appends to lid0's
          } else {
            /* add gid1 to list of ghost deleted by me -- I need their children */
            PetscMPIInt proc = lghost_pe[ghost1_idx];
            PetscCheck(!lghost_matched[ghost1_idx], PETSC_COMM_SELF, PETSC_ERR_SUP, "ghost %" PetscInt_FMT " is matched", lghost_gid[ghost1_idx]);
            lghost_matched[ghost1_idx] = PETSC_TRUE;
            PetscCall(PetscCDAppendID(ghost_deleted_list, proc, ghost1_idx)); /* cache to send messages */
            PetscCall(PetscCDAppendID(ghost_deleted_list, proc, lid0));
          }
          lid_matched[lid0] = PETSC_TRUE; /* keep track of what we've done this round */
          /* set projection */
          PetscCall(MatSetValues(P, 1, &gid0, 1, &gid0, &one, INSERT_VALUES));
          PetscCall(MatSetValues(P, 1, &gid1, 1, &gid0, &one, INSERT_VALUES));
          //PetscCall(PetscPrintf(comm,"\t %" PetscInt_FMT ".%" PetscInt_FMT ") match active EDGE %" PetscInt_FMT " : (%" PetscInt_FMT " %" PetscInt_FMT ")\n",iter,sub_it, nactive_edges, gid0, gid1));
        } /* matched */
      } /* edge loop */
      PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
      if (isMPI) PetscCall(VecRestoreArrayRead(ghostMaxEdge, &lghost_max_ew));
      PetscCall(VecRestoreArrayRead(locMaxEdge, &lid_max_ew));
      // count active for test, latter, update deleted ghosts
      n_act_n[0] = nactive_edges;
      if (ghost_deleted_list) PetscCall(PetscCDCount(ghost_deleted_list, &n_act_n[2]));
      else n_act_n[2] = 0;
      PetscCall(PetscCDCount(agg_llists, &n_act_n[1]));
      PetscCallMPI(MPIU_Allreduce(n_act_n, gn_act_n, 3, MPIU_INT, MPI_SUM, comm));
      PetscCall(PetscInfo(info_is, "[%d] %" PetscInt_FMT ".%" PetscInt_FMT ") nactive edges=%" PetscInt_FMT ", ncomm_procs=%d, nEdges=%" PetscInt_FMT ", %" PetscInt_FMT " deleted ghosts, N=%" PetscInt_FMT "\n", rank, iter, sub_it, gn_act_n[0], ncomm_procs, nEdges, gn_act_n[2], gn_act_n[1]));
      /* deal with deleted ghost */
      if (isMPI) {
        PetscCDIntNd *pos;
        PetscInt     *sbuffs1[REQ_BF_SIZE], ndel;
        PetscInt     *sbuffs2[REQ_BF_SIZE];
        MPI_Status    status;

        /* send deleted ghosts */
        for (PetscInt proc_idx = 0; proc_idx < ncomm_procs; proc_idx++) {
          const PetscMPIInt proc = comm_procs[proc_idx];
          PetscInt         *sbuff, *pt, scount;
          MPI_Request      *request;

          /* count ghosts */
          PetscCall(PetscCDCountAt(ghost_deleted_list, proc, &ndel));
          ndel /= 2; // two entries for each proc
          scount = 2 + 2 * ndel;
          PetscCall(PetscMalloc1(scount + request_size, &sbuff));
          /* save requests */
          sbuffs1[proc_idx] = sbuff;
          request           = (MPI_Request *)sbuff;
          sbuff = pt = sbuff + request_size;
          /* write [ndel, proc, n*[gid1,gid0] */
          *pt++ = ndel; // number of deleted to send
          *pt++ = rank; // proc (not used)
          PetscCall(PetscCDGetHeadPos(ghost_deleted_list, proc, &pos));
          while (pos) {
            PetscInt lid0, ghost_idx, gid1;

            PetscCall(PetscCDIntNdGetID(pos, &ghost_idx));
            gid1 = lghost_gid[ghost_idx];
            PetscCall(PetscCDGetNextPos(ghost_deleted_list, proc, &pos));
            PetscCall(PetscCDIntNdGetID(pos, &lid0));
            PetscCall(PetscCDGetNextPos(ghost_deleted_list, proc, &pos));
            *pt++ = gid1;
            *pt++ = lid0 + Istart; // gid0
          }
          PetscCheck(pt - sbuff == scount, PETSC_COMM_SELF, PETSC_ERR_SUP, "sbuff-pt != scount: %zu", pt - sbuff);
          /* MPIU_Isend:  tag1 [ndel, proc, n*[gid1,gid0] ] */
          PetscCallMPI(MPIU_Isend(sbuff, scount, MPIU_INT, proc, tag1, comm, request));
          PetscCall(PetscCDRemoveAllAt(ghost_deleted_list, proc)); // done with this list
        }
        /* receive deleted, send back partial aggregates, clear lists */
        for (PetscInt proc_idx = 0; proc_idx < ncomm_procs; proc_idx++) {
          PetscCallMPI(MPI_Probe(comm_procs[proc_idx] /* MPI_ANY_SOURCE */, tag1, comm, &status));
          {
            PetscInt         *pt, *pt2, *pt3, *sbuff, tmp;
            MPI_Request      *request;
            PetscMPIInt       rcount, scount;
            const PetscMPIInt proc = status.MPI_SOURCE;

            PetscCallMPI(MPI_Get_count(&status, MPIU_INT, &rcount));
            if (rcount > rbuff_sz) {
              if (rbuff) PetscCall(PetscFree(rbuff));
              PetscCall(PetscMalloc1(rcount, &rbuff));
              rbuff_sz = rcount;
            }
            /* MPI_Recv: tag1 [ndel, proc, ndel*[gid1,gid0] ] */
            PetscCallMPI(MPI_Recv(rbuff, rcount, MPIU_INT, proc, tag1, comm, &status));
            /* read and count sends *[lid0, n, n*[gid] ] */
            pt     = rbuff;
            scount = 0;
            ndel   = *pt++; // number of deleted to recv
            tmp    = *pt++; // proc (not used)
            while (ndel--) {
              PetscInt gid1 = *pt++, lid1 = gid1 - Istart;
              PetscInt gh_gid0 = *pt++; // gid on other proc (not used here to count)

              PetscCheck(lid1 >= 0 && lid1 < nloc, PETSC_COMM_SELF, PETSC_ERR_SUP, "received ghost deleted %" PetscInt_FMT, gid1);
              PetscCheck(!lid_matched[lid1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "%" PetscInt_FMT ") received matched local gid %" PetscInt_FMT ",%" PetscInt_FMT ", with ghost (lid) %" PetscInt_FMT " from proc %d", sub_it, gid1, gh_gid0, tmp, proc);
              lid_matched[lid1] = PETSC_TRUE;                    /* keep track of what we've done this round */
              PetscCall(PetscCDCountAt(agg_llists, lid1, &tmp)); // n
              scount += tmp + 2;                                 // lid0, n, n*[gid]
            }
            PetscCheck((pt - rbuff) == (ptrdiff_t)rcount, PETSC_COMM_SELF, PETSC_ERR_SUP, "receive buffer size != num read: %zu; rcount: %d", pt - rbuff, rcount);
            /* send tag2: *[gid0, n, n*[gid] ] */
            PetscCall(PetscMalloc1(scount + request_size, &sbuff));
            sbuffs2[proc_idx] = sbuff; /* cache request */
            request           = (MPI_Request *)sbuff;
            pt2 = sbuff = sbuff + request_size;
            // read again: n, proc, n*[gid1,gid0]
            pt   = rbuff;
            ndel = *pt++;
            tmp  = *pt++; // proc (not used)
            while (ndel--) {
              PetscInt gid1 = *pt++, lid1 = gid1 - Istart, gh_gid0 = *pt++;

              /* write [gid0, aggSz, aggSz[gid] ] */
              *pt2++ = gh_gid0;
              pt3    = pt2++; /* save pointer for later */
              PetscCall(PetscCDGetHeadPos(agg_llists, lid1, &pos));
              while (pos) {
                PetscInt gid;

                PetscCall(PetscCDIntNdGetID(pos, &gid));
                PetscCall(PetscCDGetNextPos(agg_llists, lid1, &pos));
                *pt2++ = gid;
              }
              PetscCall(PetscIntCast(pt2 - pt3 - 1, pt3));
              /* clear list */
              PetscCall(PetscCDRemoveAllAt(agg_llists, lid1));
            }
            PetscCheck((pt2 - sbuff) == (ptrdiff_t)scount, PETSC_COMM_SELF, PETSC_ERR_SUP, "buffer size != num write: %zu %d", pt2 - sbuff, scount);
            /* MPIU_Isend: requested data tag2 *[lid0, n, n*[gid1] ] */
            PetscCallMPI(MPIU_Isend(sbuff, scount, MPIU_INT, proc, tag2, comm, request));
          }
        } // proc_idx
        /* receive tag2 *[gid0, n, n*[gid] ] */
        for (PetscMPIInt proc_idx = 0; proc_idx < ncomm_procs; proc_idx++) {
          PetscMPIInt proc;
          PetscInt   *pt;
          int         rcount;

          PetscCallMPI(MPI_Probe(comm_procs[proc_idx] /* MPI_ANY_SOURCE */, tag2, comm, &status));
          PetscCallMPI(MPI_Get_count(&status, MPIU_INT, &rcount));
          if (rcount > rbuff_sz) {
            if (rbuff) PetscCall(PetscFree(rbuff));
            PetscCall(PetscMalloc1(rcount, &rbuff));
            rbuff_sz = rcount;
          }
          proc = status.MPI_SOURCE;
          /* MPI_Recv:  tag1 [n, proc, n*[gid1,lid0] ] */
          PetscCallMPI(MPI_Recv(rbuff, rcount, MPIU_INT, proc, tag2, comm, &status));
          pt = rbuff;
          while (pt - rbuff < rcount) {
            PetscInt gid0 = *pt++, n = *pt++;

            while (n--) {
              PetscInt gid1 = *pt++;

              PetscCall(PetscCDAppendID(agg_llists, gid0 - Istart, gid1));
            }
          }
          PetscCheck((pt - rbuff) == (ptrdiff_t)rcount, PETSC_COMM_SELF, PETSC_ERR_SUP, "recv buffer size != num read: %zu %d", pt - rbuff, rcount);
        }
        /* wait for tag1 isends */
        for (PetscMPIInt proc_idx = 0; proc_idx < ncomm_procs; proc_idx++) {
          MPI_Request *request = (MPI_Request *)sbuffs1[proc_idx];

          PetscCallMPI(MPI_Wait(request, &status));
          PetscCall(PetscFree(sbuffs1[proc_idx]));
        }
        /* wait for tag2 isends */
        for (PetscMPIInt proc_idx = 0; proc_idx < ncomm_procs; proc_idx++) {
          MPI_Request *request = (MPI_Request *)sbuffs2[proc_idx];

          PetscCallMPI(MPI_Wait(request, &status));
          PetscCall(PetscFree(sbuffs2[proc_idx]));
        }
      } /* MPI */
      /* set 'lghost_matched' - use locMaxEdge, ghostMaxEdge (recomputed next) */
      if (isMPI) {
        const PetscScalar *sbuff;

        for (PetscInt kk = 0, gid = Istart; kk < nloc; kk++, gid++) {
          PetscScalar vval = lid_matched[kk] ? 1.0 : 0.0;

          PetscCall(VecSetValues(locMaxEdge, 1, &gid, &vval, INSERT_VALUES)); /* set with GID */
        }
        PetscCall(VecAssemblyBegin(locMaxEdge));
        PetscCall(VecAssemblyEnd(locMaxEdge));
        PetscCall(VecScatterBegin(mpimat->Mvctx, locMaxEdge, ghostMaxEdge, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(mpimat->Mvctx, locMaxEdge, ghostMaxEdge, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecGetArrayRead(ghostMaxEdge, &sbuff));
        for (PetscInt kk = 0; kk < num_ghosts; kk++) lghost_matched[kk] = (PetscBool)(PetscRealPart(sbuff[kk]) != 0.0);
        PetscCall(VecRestoreArrayRead(ghostMaxEdge, &sbuff));
      }
      /* compute 'locMaxEdge' inside sub iteration b/c max weight can drop as neighbors are matched */
      for (PetscInt kk = 0, gid = Istart; kk < nloc; kk++, gid++) {
        PetscReal      max_e = 0., tt;
        PetscScalar    vval;
        const PetscInt lid    = kk;
        PetscMPIInt    max_pe = rank, pe, n;

        ii = matA->i;
        PetscCall(PetscMPIIntCast(ii[lid + 1] - ii[lid], &n));
        aj = PetscSafePointerPlusOffset(matA->j, ii[lid]);
        ap = PetscSafePointerPlusOffset(matA->a, ii[lid]);
        for (PetscMPIInt jj = 0; jj < n; jj++) {
          PetscInt lidj = aj[jj];

          if (lid_matched[lidj]) continue; /* this is new - can change local max */
          if (lidj != lid && PetscRealPart(ap[jj]) > max_e) max_e = PetscRealPart(ap[jj]);
        }
        if (lid_cprowID && (ix = lid_cprowID[lid]) != -1) { /* if I have any ghost neighbors */
          ii = matB->compressedrow.i;
          PetscCall(PetscMPIIntCast(ii[ix + 1] - ii[ix], &n));
          ap = matB->a + ii[ix];
          aj = matB->j + ii[ix];
          for (PetscMPIInt jj = 0; jj < n; jj++) {
            PetscInt lidj = aj[jj];

            if (lghost_matched[lidj]) continue;
            if ((tt = PetscRealPart(ap[jj])) > max_e) max_e = tt;
          }
        }
        vval = max_e;
        PetscCall(VecSetValues(locMaxEdge, 1, &gid, &vval, INSERT_VALUES)); /* set with GID */
        // max PE with max edge
        if (lid_cprowID && (ix = lid_cprowID[lid]) != -1) { /* if I have any ghost neighbors */
          ii = matB->compressedrow.i;
          PetscCall(PetscMPIIntCast(ii[ix + 1] - ii[ix], &n));
          ap = matB->a + ii[ix];
          aj = matB->j + ii[ix];
          for (PetscInt jj = 0; jj < n; jj++) {
            PetscInt lidj = aj[jj];

            if (lghost_matched[lidj]) continue;
            if ((pe = lghost_pe[aj[jj]]) > max_pe && PetscRealPart(ap[jj]) >= max_e - MY_MEPS) max_pe = pe;
          }
        }
        vval = max_pe;
        PetscCall(VecSetValues(locMaxPE, 1, &gid, &vval, INSERT_VALUES));
      }
      PetscCall(VecAssemblyBegin(locMaxEdge));
      PetscCall(VecAssemblyEnd(locMaxEdge));
      PetscCall(VecAssemblyBegin(locMaxPE));
      PetscCall(VecAssemblyEnd(locMaxPE));
      /* compute 'lghost_max_ew' and 'lghost_max_pe' to get ready for next iteration*/
      if (isMPI) {
        const PetscScalar *buf;

        PetscCall(VecScatterBegin(mpimat->Mvctx, locMaxEdge, ghostMaxEdge, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(mpimat->Mvctx, locMaxEdge, ghostMaxEdge, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterBegin(mpimat->Mvctx, locMaxPE, ghostMaxPE, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(mpimat->Mvctx, locMaxPE, ghostMaxPE, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecGetArrayRead(ghostMaxPE, &buf));
        for (PetscInt kk = 0; kk < num_ghosts; kk++) lghost_max_pe[kk] = (PetscMPIInt)PetscRealPart(buf[kk]); // the MAX proc of the ghost now
        PetscCall(VecRestoreArrayRead(ghostMaxPE, &buf));
      }
      // if no active edges, stop
      if (gn_act_n[0] < 1) break;
      // inc and check (self stopping iteration
      PetscCheck(old_num_edge != gn_act_n[0], PETSC_COMM_SELF, PETSC_ERR_SUP, "HEM stalled step %" PetscInt_FMT "/%" PetscInt_FMT, sub_it + 1, n_sub_its);
      sub_it++;
      PetscCheck(sub_it < n_sub_its, PETSC_COMM_SELF, PETSC_ERR_SUP, "failed to finish HEM step %" PetscInt_FMT "/%" PetscInt_FMT, sub_it + 1, n_sub_its);
      old_num_edge = gn_act_n[0];
    } /* sub_it loop */
    /* clean up iteration */
    PetscCall(PetscFree(Edges));
    if (isMPI) { // can be hoisted
      PetscCall(VecRestoreArrayRead(ghostMaxEdge, &lghost_max_ew));
      PetscCall(VecDestroy(&ghostMaxEdge));
      PetscCall(VecDestroy(&ghostMaxPE));
      PetscCall(PetscFree4(lghost_matched, lghost_pe, lghost_gid, lghost_max_pe));
    }
    PetscCall(VecDestroy(&locMaxEdge));
    PetscCall(VecDestroy(&locMaxPE));
    /* create next graph */
    {
      Vec diag;

      /* add identity for unmatched vertices so they stay alive */
      for (PetscInt kk = 0, gid1, gid = Istart; kk < nloc; kk++, gid++) {
        if (!lid_matched[kk]) {
          const PetscInt lid = kk;
          PetscCDIntNd  *pos;

          PetscCall(PetscCDGetHeadPos(agg_llists, lid, &pos));
          PetscCheck(pos, PETSC_COMM_SELF, PETSC_ERR_PLIB, "empty list in singleton: %" PetscInt_FMT, gid);
          PetscCall(PetscCDIntNdGetID(pos, &gid1));
          PetscCheck(gid1 == gid, PETSC_COMM_SELF, PETSC_ERR_PLIB, "first in list (%" PetscInt_FMT ") in singleton not %" PetscInt_FMT, gid1, gid);
          PetscCall(MatSetValues(P, 1, &gid, 1, &gid, &one, INSERT_VALUES));
        }
      }
      PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

      /* project to make new graph with collapsed edges */
      PetscCall(MatPtAP(cMat, P, MAT_INITIAL_MATRIX, 1.0, &tMat));
      PetscCall(MatDestroy(&P));
      PetscCall(MatDestroy(&cMat));
      cMat = tMat;
      PetscCall(MatCreateVecs(cMat, &diag, NULL));
      PetscCall(MatGetDiagonal(cMat, diag));
      PetscCall(VecReciprocal(diag));
      PetscCall(VecSqrtAbs(diag));
      PetscCall(MatDiagonalScale(cMat, diag, diag));
      PetscCall(VecDestroy(&diag));
    }
  } /* coarsen iterator */

  /* make fake matrix with Mat->B only for smoothed agg QR. Need this if we make an aux graph (ie, PtAP) with k > 1 */
  if (size > 1) {
    Mat           mat;
    PetscCDIntNd *pos;
    PetscInt      NN, MM, jj = 0, mxsz = 0;

    for (PetscInt kk = 0; kk < nloc; kk++) {
      PetscCall(PetscCDCountAt(agg_llists, kk, &jj));
      if (jj > mxsz) mxsz = jj;
    }
    PetscCall(MatGetSize(a_Gmat, &MM, &NN));
    if (mxsz > MM - nloc) mxsz = MM - nloc;
    /* matrix of ghost adj for square graph */
    PetscCall(MatCreateAIJ(comm, nloc, nloc, PETSC_DETERMINE, PETSC_DETERMINE, 0, NULL, mxsz, NULL, &mat));
    for (PetscInt lid = 0, gid = Istart; lid < nloc; lid++, gid++) {
      PetscCall(PetscCDGetHeadPos(agg_llists, lid, &pos));
      while (pos) {
        PetscInt gid1;

        PetscCall(PetscCDIntNdGetID(pos, &gid1));
        PetscCall(PetscCDGetNextPos(agg_llists, lid, &pos));
        if (gid1 < Istart || gid1 >= Istart + nloc) PetscCall(MatSetValues(mat, 1, &gid, 1, &gid1, &one, ADD_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscCDSetMat(agg_llists, mat));
    PetscCall(PetscCDDestroy(ghost_deleted_list));
    if (rbuff_sz) PetscCall(PetscFree(rbuff)); // always true
  }
  // move BCs into some node
  if (bc_list) {
    PetscCDIntNd *pos;

    PetscCall(PetscCDGetHeadPos(bc_list, 0, &pos));
    while (pos) {
      PetscInt gid1;

      PetscCall(PetscCDIntNdGetID(pos, &gid1));
      PetscCall(PetscCDGetNextPos(bc_list, 0, &pos));
      PetscCall(PetscCDAppendID(agg_llists, bc_agg, gid1));
    }
    PetscCall(PetscCDRemoveAllAt(bc_list, 0));
    PetscCall(PetscCDDestroy(bc_list));
  }
  {
    // check sizes -- all vertices must get in graph
    PetscInt sz, globalsz, MM;

    PetscCall(MatGetSize(a_Gmat, &MM, NULL));
    PetscCall(PetscCDCount(agg_llists, &sz));
    PetscCallMPI(MPIU_Allreduce(&sz, &globalsz, 1, MPIU_INT, MPI_SUM, comm));
    PetscCheck(MM == globalsz, comm, PETSC_ERR_SUP, "lost %" PetscInt_FMT " equations ?", MM - globalsz);
  }
  // cleanup
  PetscCall(MatDestroy(&cMat));
  PetscCall(PetscFree3(lid_matched, lid_cprowID, lid_max_pe));
  PetscCall(ISDestroy(&info_is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   HEM coarsen, simple greedy.
*/
static PetscErrorCode MatCoarsenApply_HEM(MatCoarsen coarse)
{
  Mat mat = coarse->graph;

  PetscFunctionBegin;
  PetscCall(MatCoarsenApply_HEM_private(mat, coarse->max_it, coarse->threshold, &coarse->agg_lists));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCoarsenView_HEM(MatCoarsen coarse, PetscViewer viewer)
{
  PetscMPIInt rank;
  PetscBool   isascii;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)coarse), &rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCDIntNd     *pos, *pos2;
    PetscViewerFormat format;

    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " matching steps with threshold = %g\n", coarse->max_it, (double)coarse->threshold));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (coarse->agg_lists) {
        PetscCall(PetscViewerASCIIPushSynchronized(viewer));
        for (PetscInt kk = 0; kk < coarse->agg_lists->size; kk++) {
          PetscCall(PetscCDGetHeadPos(coarse->agg_lists, kk, &pos));
          if ((pos2 = pos)) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "selected local %" PetscInt_FMT ": ", kk));
          while (pos) {
            PetscInt gid1;

            PetscCall(PetscCDIntNdGetID(pos, &gid1));
            PetscCall(PetscCDGetNextPos(coarse->agg_lists, kk, &pos));
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT " ", gid1));
          }
          if (pos2) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
        }
        PetscCall(PetscViewerFlush(viewer));
        PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  HEM aggregator lists are not available\n"));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATCOARSENHEM - A coarsener that uses HEM a simple greedy coarsener

   Level: beginner

.seealso: `MatCoarsen`, `MatCoarsenMISKSetDistance()`, `MatCoarsenApply()`, `MatCoarsenSetType()`, `MatCoarsenType`, `MatCoarsenCreate()`, `MATCOARSENMISK`, `MATCOARSENMIS`
M*/

PETSC_EXTERN PetscErrorCode MatCoarsenCreate_HEM(MatCoarsen coarse)
{
  PetscFunctionBegin;
  coarse->ops->apply = MatCoarsenApply_HEM;
  coarse->ops->view  = MatCoarsenView_HEM;
  coarse->max_it     = 4;
  PetscFunctionReturn(PETSC_SUCCESS);
}
