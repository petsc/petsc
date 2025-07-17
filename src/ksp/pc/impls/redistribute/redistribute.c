/*
  This file defines a "solve the problem redistributely on each subgroup of processor" preconditioner.
*/
#include <petsc/private/pcimpl.h> /*I "petscksp.h" I*/
#include <petscksp.h>

typedef struct _PC_FieldSplitLink *PC_FieldSplitLink;
struct _PC_FieldSplitLink {
  char             *splitname;
  IS                is;
  PC_FieldSplitLink next, previous;
};

typedef struct {
  KSP          ksp;
  Vec          x, b;
  VecScatter   scatter;
  IS           is;
  PetscInt     dcnt, *drows; /* these are the local rows that have only diagonal entry */
  PetscScalar *diag;
  Vec          work;
  PetscBool    zerodiag;

  PetscInt          nsplits;
  PC_FieldSplitLink splitlinks;
} PC_Redistribute;

static PetscErrorCode PCFieldSplitSetIS_Redistribute(PC pc, const char splitname[], IS is)
{
  PC_Redistribute   *red  = (PC_Redistribute *)pc->data;
  PC_FieldSplitLink *next = &red->splitlinks;

  PetscFunctionBegin;
  while (*next) next = &(*next)->next;
  PetscCall(PetscNew(next));
  if (splitname) {
    PetscCall(PetscStrallocpy(splitname, &(*next)->splitname));
  } else {
    PetscCall(PetscMalloc1(8, &(*next)->splitname));
    PetscCall(PetscSNPrintf((*next)->splitname, 7, "%" PetscInt_FMT, red->nsplits++));
  }
  PetscCall(PetscObjectReference((PetscObject)is));
  PetscCall(ISDestroy(&(*next)->is));
  (*next)->is = is;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_Redistribute(PC pc, PetscViewer viewer)
{
  PC_Redistribute *red = (PC_Redistribute *)pc->data;
  PetscBool        isascii, isstring;
  PetscInt         ncnt, N;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCallMPI(MPIU_Allreduce(&red->dcnt, &ncnt, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)pc)));
    PetscCall(MatGetSize(pc->pmat, &N, NULL));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    Number rows eliminated %" PetscInt_FMT " Percentage rows eliminated %g\n", ncnt, (double)(100 * ((PetscReal)ncnt) / ((PetscReal)N))));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Redistribute preconditioner: \n"));
    PetscCall(KSPView(red->ksp, viewer));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer, " Redistribute preconditioner"));
    PetscCall(KSPView(red->ksp, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Redistribute(PC pc)
{
  PC_Redistribute         *red = (PC_Redistribute *)pc->data;
  MPI_Comm                 comm;
  PetscInt                 rstart, rend, nrstart, nrend, nz, cnt, *rows, ncnt, dcnt, *drows;
  PetscLayout              map, nmap;
  PetscMPIInt              size, tag, n;
  PETSC_UNUSED PetscMPIInt imdex;
  PetscInt                *source = NULL;
  PetscMPIInt             *sizes  = NULL, nrecvs, nsends;
  PetscInt                 j;
  PetscInt                *owner = NULL, *starts = NULL, count, slen;
  PetscInt                *rvalues, *svalues, recvtotal;
  PetscMPIInt             *onodes1, *olengths1;
  MPI_Request             *send_waits = NULL, *recv_waits = NULL;
  MPI_Status               recv_status, *send_status;
  Vec                      tvec, diag;
  Mat                      tmat;
  const PetscScalar       *d, *values;
  const PetscInt          *cols;
  PC_FieldSplitLink       *next = &red->splitlinks;

  PetscFunctionBegin;
  if (pc->setupcalled) {
    PetscCheck(pc->flag == SAME_NONZERO_PATTERN, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "PC is not supported for a change in the nonzero structure of the matrix");
    PetscCall(KSPGetOperators(red->ksp, NULL, &tmat));
    PetscCall(MatCreateSubMatrix(pc->pmat, red->is, red->is, MAT_REUSE_MATRIX, &tmat));
    PetscCall(KSPSetOperators(red->ksp, tmat, tmat));
  } else {
    PetscInt  NN;
    PC        ipc;
    PetscBool fptr;

    PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCall(PetscObjectGetNewTag((PetscObject)pc, &tag));

    /* count non-diagonal rows on process */
    PetscCall(MatGetOwnershipRange(pc->mat, &rstart, &rend));
    cnt = 0;
    for (PetscInt i = rstart; i < rend; i++) {
      PetscCall(MatGetRow(pc->mat, i, &nz, &cols, &values));
      for (PetscInt j = 0; j < nz; j++) {
        if (values[j] != 0 && cols[j] != i) {
          cnt++;
          break;
        }
      }
      PetscCall(MatRestoreRow(pc->mat, i, &nz, &cols, &values));
    }
    PetscCall(PetscMalloc1(cnt, &rows));
    PetscCall(PetscMalloc1(rend - rstart - cnt, &drows));

    /* list non-diagonal rows on process */
    cnt  = 0;
    dcnt = 0;
    for (PetscInt i = rstart; i < rend; i++) {
      PetscBool diagonly = PETSC_TRUE;
      PetscCall(MatGetRow(pc->mat, i, &nz, &cols, &values));
      for (PetscInt j = 0; j < nz; j++) {
        if (values[j] != 0 && cols[j] != i) {
          diagonly = PETSC_FALSE;
          break;
        }
      }
      if (!diagonly) rows[cnt++] = i;
      else drows[dcnt++] = i - rstart;
      PetscCall(MatRestoreRow(pc->mat, i, &nz, &cols, &values));
    }

    /* create PetscLayout for non-diagonal rows on each process */
    PetscCall(PetscLayoutCreate(comm, &map));
    PetscCall(PetscLayoutSetLocalSize(map, cnt));
    PetscCall(PetscLayoutSetBlockSize(map, 1));
    PetscCall(PetscLayoutSetUp(map));
    nrstart = map->rstart;
    nrend   = map->rend;

    /* create PetscLayout for load-balanced non-diagonal rows on each process */
    PetscCall(PetscLayoutCreate(comm, &nmap));
    PetscCallMPI(MPIU_Allreduce(&cnt, &ncnt, 1, MPIU_INT, MPI_SUM, comm));
    PetscCall(PetscLayoutSetSize(nmap, ncnt));
    PetscCall(PetscLayoutSetBlockSize(nmap, 1));
    PetscCall(PetscLayoutSetUp(nmap));

    PetscCall(MatGetSize(pc->pmat, &NN, NULL));
    PetscCall(PetscInfo(pc, "Number of diagonal rows eliminated %" PetscInt_FMT ", percentage eliminated %g\n", NN - ncnt, (double)((PetscReal)(NN - ncnt) / (PetscReal)NN)));

    if (size > 1) {
      /*
        the following block of code assumes MPI can send messages to self, which is not supported for MPI-uni hence we need to handle
        the size 1 case as a special case

       this code is taken from VecScatterCreate_PtoS()
       Determines what rows need to be moved where to
       load balance the non-diagonal rows
       */
      /*  count number of contributors to each processor */
      PetscCall(PetscMalloc2(size, &sizes, cnt, &owner));
      PetscCall(PetscArrayzero(sizes, size));
      j      = 0;
      nsends = 0;
      for (PetscInt i = nrstart; i < nrend; i++) {
        if (i < nmap->range[j]) j = 0;
        for (; j < size; j++) {
          if (i < nmap->range[j + 1]) {
            if (!sizes[j]++) nsends++;
            owner[i - nrstart] = j;
            break;
          }
        }
      }
      /* inform other processors of number of messages and max length*/
      PetscCall(PetscGatherNumberOfMessages(comm, NULL, sizes, &nrecvs));
      PetscCall(PetscGatherMessageLengths(comm, nsends, nrecvs, sizes, &onodes1, &olengths1));
      PetscCall(PetscSortMPIIntWithArray(nrecvs, onodes1, olengths1));
      recvtotal = 0;
      for (PetscMPIInt i = 0; i < nrecvs; i++) recvtotal += olengths1[i];

      /* post receives:  rvalues - rows I will own; count - nu */
      PetscCall(PetscMalloc3(recvtotal, &rvalues, nrecvs, &source, nrecvs, &recv_waits));
      count = 0;
      for (PetscMPIInt i = 0; i < nrecvs; i++) {
        PetscCallMPI(MPIU_Irecv(rvalues + count, olengths1[i], MPIU_INT, onodes1[i], tag, comm, recv_waits + i));
        count += olengths1[i];
      }

      /* do sends:
       1) starts[i] gives the starting index in svalues for stuff going to
       the ith processor
       */
      PetscCall(PetscMalloc3(cnt, &svalues, nsends, &send_waits, size, &starts));
      starts[0] = 0;
      for (PetscMPIInt i = 1; i < size; i++) starts[i] = starts[i - 1] + sizes[i - 1];
      for (PetscInt i = 0; i < cnt; i++) svalues[starts[owner[i]]++] = rows[i];
      for (PetscInt i = 0; i < cnt; i++) rows[i] = rows[i] - nrstart;
      red->drows = drows;
      red->dcnt  = dcnt;
      PetscCall(PetscFree(rows));

      starts[0] = 0;
      for (PetscMPIInt i = 1; i < size; i++) starts[i] = starts[i - 1] + sizes[i - 1];
      count = 0;
      for (PetscMPIInt i = 0; i < size; i++) {
        if (sizes[i]) PetscCallMPI(MPIU_Isend(svalues + starts[i], sizes[i], MPIU_INT, i, tag, comm, send_waits + count++));
      }

      /*  wait on receives */
      count = nrecvs;
      slen  = 0;
      while (count) {
        PetscCallMPI(MPI_Waitany(nrecvs, recv_waits, &imdex, &recv_status));
        /* unpack receives into our local space */
        PetscCallMPI(MPI_Get_count(&recv_status, MPIU_INT, &n));
        slen += n;
        count--;
      }
      PetscCheck(slen == recvtotal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total message lengths %" PetscInt_FMT " not expected %" PetscInt_FMT, slen, recvtotal);
      PetscCall(ISCreateGeneral(comm, slen, rvalues, PETSC_COPY_VALUES, &red->is));

      /* free all work space */
      PetscCall(PetscFree(olengths1));
      PetscCall(PetscFree(onodes1));
      PetscCall(PetscFree3(rvalues, source, recv_waits));
      PetscCall(PetscFree2(sizes, owner));
      if (nsends) { /* wait on sends */
        PetscCall(PetscMalloc1(nsends, &send_status));
        PetscCallMPI(MPI_Waitall(nsends, send_waits, send_status));
        PetscCall(PetscFree(send_status));
      }
      PetscCall(PetscFree3(svalues, send_waits, starts));
    } else {
      PetscCall(ISCreateGeneral(comm, cnt, rows, PETSC_OWN_POINTER, &red->is));
      red->drows = drows;
      red->dcnt  = dcnt;
      slen       = cnt;
    }
    PetscCall(PetscLayoutDestroy(&map));

    PetscCall(VecCreateMPI(comm, slen, PETSC_DETERMINE, &red->b));
    PetscCall(VecDuplicate(red->b, &red->x));
    PetscCall(MatCreateVecs(pc->pmat, &tvec, NULL));
    PetscCall(VecScatterCreate(tvec, red->is, red->b, NULL, &red->scatter));

    /* Map the PCFIELDSPLIT fields to redistributed KSP */
    PetscCall(KSPGetPC(red->ksp, &ipc));
    PetscCall(PetscObjectHasFunction((PetscObject)ipc, "PCFieldSplitSetIS_C", &fptr));
    if (fptr && *next) {
      PetscScalar       *atvec;
      const PetscScalar *ab;
      PetscInt           primes[] = {2, 3, 5, 7, 11, 13, 17, 19};
      PetscInt           cnt      = 0;

      PetscCheck(red->nsplits <= (PetscInt)PETSC_STATIC_ARRAY_LENGTH(primes), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No support for this many fields");
      PetscCall(VecSet(tvec, 1.0));
      PetscCall(VecGetArray(tvec, &atvec));

      while (*next) {
        const PetscInt *indices;
        PetscInt        n;

        PetscCall(ISGetIndices((*next)->is, &indices));
        PetscCall(ISGetLocalSize((*next)->is, &n));
        for (PetscInt i = 0; i < n; i++) atvec[indices[i] - rstart] *= primes[cnt];
        PetscCall(ISRestoreIndices((*next)->is, &indices));
        cnt++;
        next = &(*next)->next;
      }
      PetscCall(VecRestoreArray(tvec, &atvec));
      PetscCall(VecScatterBegin(red->scatter, tvec, red->b, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(red->scatter, tvec, red->b, INSERT_VALUES, SCATTER_FORWARD));
      cnt = 0;
      PetscCall(VecGetArrayRead(red->b, &ab));
      next = &red->splitlinks;
      while (*next) {
        PetscInt  n = 0;
        PetscInt *indices;
        IS        ris;

        for (PetscInt i = 0; i < nmap->rend - nmap->rstart; i++) {
          if (!(((PetscInt)PetscRealPart(ab[i])) % primes[cnt])) n++;
        }
        PetscCall(PetscMalloc1(n, &indices));
        n = 0;
        for (PetscInt i = 0; i < nmap->rend - nmap->rstart; i++) {
          if (!(((PetscInt)PetscRealPart(ab[i])) % primes[cnt])) indices[n++] = i + nmap->rstart;
        }
        PetscCall(ISCreateGeneral(comm, n, indices, PETSC_OWN_POINTER, &ris));
        PetscCall(PCFieldSplitSetIS(ipc, (*next)->splitname, ris));

        PetscCall(ISDestroy(&ris));
        cnt++;
        next = &(*next)->next;
      }
      PetscCall(VecRestoreArrayRead(red->b, &ab));
    }
    PetscCall(VecDestroy(&tvec));
    PetscCall(MatCreateSubMatrix(pc->pmat, red->is, red->is, MAT_INITIAL_MATRIX, &tmat));
    PetscCall(KSPSetOperators(red->ksp, tmat, tmat));
    PetscCall(MatDestroy(&tmat));
    PetscCall(PetscLayoutDestroy(&nmap));
  }

  /* get diagonal portion of matrix */
  PetscCall(PetscFree(red->diag));
  PetscCall(PetscMalloc1(red->dcnt, &red->diag));
  PetscCall(MatCreateVecs(pc->pmat, &diag, NULL));
  PetscCall(MatGetDiagonal(pc->pmat, diag));
  PetscCall(VecGetArrayRead(diag, &d));
  for (PetscInt i = 0; i < red->dcnt; i++) {
    if (d[red->drows[i]] != 0) red->diag[i] = 1.0 / d[red->drows[i]];
    else {
      red->zerodiag = PETSC_TRUE;
      red->diag[i]  = 0.0;
    }
  }
  PetscCall(VecRestoreArrayRead(diag, &d));
  PetscCall(VecDestroy(&diag));
  PetscCall(KSPSetUp(red->ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_Redistribute(PC pc, Vec b, Vec x)
{
  PC_Redistribute   *red   = (PC_Redistribute *)pc->data;
  PetscInt           dcnt  = red->dcnt, i;
  const PetscInt    *drows = red->drows;
  PetscScalar       *xwork;
  const PetscScalar *bwork, *diag = red->diag;
  PetscBool          nonzero_guess;

  PetscFunctionBegin;
  if (!red->work) PetscCall(VecDuplicate(b, &red->work));
  PetscCall(KSPGetInitialGuessNonzero(red->ksp, &nonzero_guess));
  if (nonzero_guess) {
    PetscCall(VecScatterBegin(red->scatter, x, red->x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(red->scatter, x, red->x, INSERT_VALUES, SCATTER_FORWARD));
  }

  /* compute the rows of solution that have diagonal entries only */
  PetscCall(VecSet(x, 0.0)); /* x = diag(A)^{-1} b */
  PetscCall(VecGetArray(x, &xwork));
  PetscCall(VecGetArrayRead(b, &bwork));
  if (red->zerodiag) {
    for (i = 0; i < dcnt; i++) {
      if (diag[i] == 0.0 && bwork[drows[i]] != 0.0) {
        PetscCheck(!pc->erroriffailure, PETSC_COMM_SELF, PETSC_ERR_CONV_FAILED, "Linear system is inconsistent, zero matrix row but nonzero right-hand side");
        PetscCall(PetscInfo(pc, "Linear system is inconsistent, zero matrix row but nonzero right-hand side\n"));
        pc->failedreasonrank = PC_INCONSISTENT_RHS;
      }
    }
    PetscCall(VecFlag(x, pc->failedreasonrank == PC_INCONSISTENT_RHS));
  }
  for (i = 0; i < dcnt; i++) xwork[drows[i]] = diag[i] * bwork[drows[i]];
  PetscCall(PetscLogFlops(dcnt));
  PetscCall(VecRestoreArray(red->work, &xwork));
  PetscCall(VecRestoreArrayRead(b, &bwork));
  /* update the right-hand side for the reduced system with diagonal rows (and corresponding columns) removed */
  PetscCall(MatMult(pc->pmat, x, red->work));
  PetscCall(VecAYPX(red->work, -1.0, b)); /* red->work = b - A x */

  PetscCall(VecScatterBegin(red->scatter, red->work, red->b, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(red->scatter, red->work, red->b, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(KSPSolve(red->ksp, red->b, red->x));
  PetscCall(KSPCheckSolve(red->ksp, pc, red->x));
  PetscCall(VecScatterBegin(red->scatter, red->x, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(red->scatter, red->x, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_Redistribute(PC pc, Vec b, Vec x)
{
  PC_Redistribute   *red   = (PC_Redistribute *)pc->data;
  PetscInt           dcnt  = red->dcnt, i;
  const PetscInt    *drows = red->drows;
  PetscScalar       *xwork;
  const PetscScalar *bwork, *diag = red->diag;
  PetscBool          set, flg     = PETSC_FALSE, nonzero_guess;

  PetscFunctionBegin;
  PetscCall(MatIsStructurallySymmetricKnown(pc->pmat, &set, &flg));
  PetscCheck(set || flg, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "PCApplyTranspose() not implemented for structurally unsymmetric Mat");
  if (!red->work) PetscCall(VecDuplicate(b, &red->work));
  PetscCall(KSPGetInitialGuessNonzero(red->ksp, &nonzero_guess));
  if (nonzero_guess) {
    PetscCall(VecScatterBegin(red->scatter, x, red->x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(red->scatter, x, red->x, INSERT_VALUES, SCATTER_FORWARD));
  }

  /* compute the rows of solution that have diagonal entries only */
  PetscCall(VecSet(x, 0.0)); /* x = diag(A)^{-1} b */
  PetscCall(VecGetArray(x, &xwork));
  PetscCall(VecGetArrayRead(b, &bwork));
  if (red->zerodiag) {
    for (i = 0; i < dcnt; i++) {
      if (diag[i] == 0.0 && bwork[drows[i]] != 0.0) {
        PetscCheck(!pc->erroriffailure, PETSC_COMM_SELF, PETSC_ERR_CONV_FAILED, "Linear system is inconsistent, zero matrix row but nonzero right-hand side");
        PetscCall(PetscInfo(pc, "Linear system is inconsistent, zero matrix row but nonzero right-hand side\n"));
        pc->failedreasonrank = PC_INCONSISTENT_RHS;
      }
    }
    PetscCall(VecFlag(x, pc->failedreasonrank == PC_INCONSISTENT_RHS));
  }
  for (i = 0; i < dcnt; i++) xwork[drows[i]] = diag[i] * bwork[drows[i]];
  PetscCall(PetscLogFlops(dcnt));
  PetscCall(VecRestoreArray(red->work, &xwork));
  PetscCall(VecRestoreArrayRead(b, &bwork));
  /* update the right-hand side for the reduced system with diagonal rows (and corresponding columns) removed */
  PetscCall(MatMultTranspose(pc->pmat, x, red->work));
  PetscCall(VecAYPX(red->work, -1.0, b)); /* red->work = b - A^T x */

  PetscCall(VecScatterBegin(red->scatter, red->work, red->b, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(red->scatter, red->work, red->b, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(KSPSolveTranspose(red->ksp, red->b, red->x));
  PetscCall(KSPCheckSolve(red->ksp, pc, red->x));
  PetscCall(VecScatterBegin(red->scatter, red->x, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(red->scatter, red->x, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Redistribute(PC pc)
{
  PC_Redistribute  *red  = (PC_Redistribute *)pc->data;
  PC_FieldSplitLink next = red->splitlinks;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFieldSplitSetIS_C", NULL));

  while (next) {
    PC_FieldSplitLink ilink;
    PetscCall(PetscFree(next->splitname));
    PetscCall(ISDestroy(&next->is));
    ilink = next;
    next  = next->next;
    PetscCall(PetscFree(ilink));
  }
  PetscCall(VecScatterDestroy(&red->scatter));
  PetscCall(ISDestroy(&red->is));
  PetscCall(VecDestroy(&red->b));
  PetscCall(VecDestroy(&red->x));
  PetscCall(KSPDestroy(&red->ksp));
  PetscCall(VecDestroy(&red->work));
  PetscCall(PetscFree(red->drows));
  PetscCall(PetscFree(red->diag));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_Redistribute(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_Redistribute *red = (PC_Redistribute *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPSetFromOptions(red->ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCRedistributeGetKSP - Gets the `KSP` created by the `PCREDISTRIBUTE`

  Not Collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. innerksp - the inner `KSP`

  Level: advanced

.seealso: [](ch_ksp), `KSP`, `PCREDISTRIBUTE`
@*/
PetscErrorCode PCRedistributeGetKSP(PC pc, KSP *innerksp)
{
  PC_Redistribute *red = (PC_Redistribute *)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(innerksp, 2);
  *innerksp = red->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCREDISTRIBUTE - Redistributes a matrix for load balancing, removing the rows (and the corresponding columns) that only have a diagonal entry and then
     applies a `KSP` to that new smaller matrix

     Level: intermediate

     Notes:
     Options for the redistribute `KSP` and `PC` with the options database prefix `-redistribute_`

     Usually run this with `-ksp_type preonly`

     If you have used `MatZeroRows()` to eliminate (for example, Dirichlet) boundary conditions for a symmetric problem then you can use, for example, `-ksp_type preonly
     -pc_type redistribute -redistribute_ksp_type cg -redistribute_pc_type bjacobi -redistribute_sub_pc_type icc` to take advantage of the symmetry.

     Supports the function `PCFieldSplitSetIS()`; pass the appropriate reduced field indices to an inner `PCFIELDSPLIT`, set with, for example
     `-ksp_type preonly -pc_type redistribute -redistribute_pc_type fieldsplit`. Does not support the `PCFIELDSPLIT` options database keys.

     This does NOT call a partitioner to reorder rows to lower communication; the ordering of the rows in the original matrix and redistributed matrix is the same. Rows are moved
     between MPI processes inside the preconditioner to balance the number of rows on each process.

     The matrix block information is lost with the possible removal of individual rows and columns of the matrix, thus the behavior of the preconditioner on the reduced
     system may be very different (worse) than running that preconditioner on the full system. This is specifically true for elasticity problems.

     Developer Note:
     Should add an option to this preconditioner to use a partitioner to redistribute the rows to lower communication.

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCType`, `PCRedistributeGetKSP()`, `MatZeroRows()`, `PCFieldSplitSetIS()`, `PCFIELDSPLIT`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Redistribute(PC pc)
{
  PC_Redistribute *red;
  const char      *prefix;

  PetscFunctionBegin;
  PetscCall(PetscNew(&red));
  pc->data = (void *)red;

  pc->ops->apply          = PCApply_Redistribute;
  pc->ops->applytranspose = PCApplyTranspose_Redistribute;
  pc->ops->setup          = PCSetUp_Redistribute;
  pc->ops->destroy        = PCDestroy_Redistribute;
  pc->ops->setfromoptions = PCSetFromOptions_Redistribute;
  pc->ops->view           = PCView_Redistribute;

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &red->ksp));
  PetscCall(KSPSetNestLevel(red->ksp, pc->kspnestlevel));
  PetscCall(KSPSetErrorIfNotConverged(red->ksp, pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)red->ksp, (PetscObject)pc, 1));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(KSPSetOptionsPrefix(red->ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(red->ksp, "redistribute_"));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFieldSplitSetIS_C", PCFieldSplitSetIS_Redistribute));
  PetscFunctionReturn(PETSC_SUCCESS);
}
