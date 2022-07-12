
/*
  This file defines a "solve the problem redistributely on each subgroup of processor" preconditioner.
*/
#include <petsc/private/pcimpl.h>     /*I "petscksp.h" I*/
#include <petscksp.h>

typedef struct {
  KSP         ksp;
  Vec         x,b;
  VecScatter  scatter;
  IS          is;
  PetscInt    dcnt,*drows;    /* these are the local rows that have only diagonal entry */
  PetscScalar *diag;
  Vec         work;
} PC_Redistribute;

static PetscErrorCode PCView_Redistribute(PC pc,PetscViewer viewer)
{
  PC_Redistribute *red = (PC_Redistribute*)pc->data;
  PetscBool       iascii,isstring;
  PetscInt        ncnt,N;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  if (iascii) {
    PetscCall(MPIU_Allreduce(&red->dcnt,&ncnt,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
    PetscCall(MatGetSize(pc->pmat,&N,NULL));
    PetscCall(PetscViewerASCIIPrintf(viewer,"    Number rows eliminated %" PetscInt_FMT " Percentage rows eliminated %g\n",ncnt,(double)(100.0*((PetscReal)ncnt)/((PetscReal)N))));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Redistribute preconditioner: \n"));
    PetscCall(KSPView(red->ksp,viewer));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer," Redistribute preconditioner"));
    PetscCall(KSPView(red->ksp,viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Redistribute(PC pc)
{
  PC_Redistribute          *red = (PC_Redistribute*)pc->data;
  MPI_Comm                 comm;
  PetscInt                 rstart,rend,i,nz,cnt,*rows,ncnt,dcnt,*drows;
  PetscLayout              map,nmap;
  PetscMPIInt              size,tag,n;
  PETSC_UNUSED PetscMPIInt imdex;
  PetscInt                 *source = NULL;
  PetscMPIInt              *sizes = NULL,nrecvs;
  PetscInt                 j,nsends;
  PetscInt                 *owner = NULL,*starts = NULL,count,slen;
  PetscInt                 *rvalues,*svalues,recvtotal;
  PetscMPIInt              *onodes1,*olengths1;
  MPI_Request              *send_waits = NULL,*recv_waits = NULL;
  MPI_Status               recv_status,*send_status;
  Vec                      tvec,diag;
  Mat                      tmat;
  const PetscScalar        *d,*values;
  const PetscInt           *cols;

  PetscFunctionBegin;
  if (pc->setupcalled) {
    PetscCall(KSPGetOperators(red->ksp,NULL,&tmat));
    PetscCall(MatCreateSubMatrix(pc->pmat,red->is,red->is,MAT_REUSE_MATRIX,&tmat));
    PetscCall(KSPSetOperators(red->ksp,tmat,tmat));
  } else {
    PetscInt NN;

    PetscCall(PetscObjectGetComm((PetscObject)pc,&comm));
    PetscCallMPI(MPI_Comm_size(comm,&size));
    PetscCall(PetscObjectGetNewTag((PetscObject)pc,&tag));

    /* count non-diagonal rows on process */
    PetscCall(MatGetOwnershipRange(pc->mat,&rstart,&rend));
    cnt  = 0;
    for (i=rstart; i<rend; i++) {
      PetscCall(MatGetRow(pc->mat,i,&nz,&cols,&values));
      for (PetscInt j=0; j<nz; j++) {
        if (values[j] != 0 && cols[j] != i) {
          cnt++;
          break;
        }
      }
      PetscCall(MatRestoreRow(pc->mat,i,&nz,&cols,&values));
    }
    PetscCall(PetscMalloc1(cnt,&rows));
    PetscCall(PetscMalloc1(rend - rstart - cnt,&drows));

    /* list non-diagonal rows on process */
    cnt = 0; dcnt = 0;
    for (i=rstart; i<rend; i++) {
      PetscBool diagonly = PETSC_TRUE;
      PetscCall(MatGetRow(pc->mat,i,&nz,&cols,&values));
      for (PetscInt j=0; j<nz; j++) {
        if (values[j] != 0 && cols[j] != i) {
          diagonly = PETSC_FALSE;
          break;
        }
      }
      if (!diagonly) rows[cnt++] = i;
      else drows[dcnt++] = i - rstart;
      PetscCall(MatRestoreRow(pc->mat,i,&nz,&cols,&values));
    }

    /* create PetscLayout for non-diagonal rows on each process */
    PetscCall(PetscLayoutCreate(comm,&map));
    PetscCall(PetscLayoutSetLocalSize(map,cnt));
    PetscCall(PetscLayoutSetBlockSize(map,1));
    PetscCall(PetscLayoutSetUp(map));
    rstart = map->rstart;
    rend   = map->rend;

    /* create PetscLayout for load-balanced non-diagonal rows on each process */
    PetscCall(PetscLayoutCreate(comm,&nmap));
    PetscCall(MPIU_Allreduce(&cnt,&ncnt,1,MPIU_INT,MPI_SUM,comm));
    PetscCall(PetscLayoutSetSize(nmap,ncnt));
    PetscCall(PetscLayoutSetBlockSize(nmap,1));
    PetscCall(PetscLayoutSetUp(nmap));

    PetscCall(MatGetSize(pc->pmat,&NN,NULL));
    PetscCall(PetscInfo(pc,"Number of diagonal rows eliminated %" PetscInt_FMT ", percentage eliminated %g\n",NN-ncnt,(double)(((PetscReal)(NN-ncnt))/((PetscReal)(NN)))));

    if (size > 1) {
      /* the following block of code assumes MPI can send messages to self, which is not supported for MPI-uni hence we need to handle the size 1 case as a special case */
      /*
       this code is taken from VecScatterCreate_PtoS()
       Determines what rows need to be moved where to
       load balance the non-diagonal rows
       */
      /*  count number of contributors to each processor */
      PetscCall(PetscMalloc2(size,&sizes,cnt,&owner));
      PetscCall(PetscArrayzero(sizes,size));
      j      = 0;
      nsends = 0;
      for (i=rstart; i<rend; i++) {
        if (i < nmap->range[j]) j = 0;
        for (; j<size; j++) {
          if (i < nmap->range[j+1]) {
            if (!sizes[j]++) nsends++;
            owner[i-rstart] = j;
            break;
          }
        }
      }
      /* inform other processors of number of messages and max length*/
      PetscCall(PetscGatherNumberOfMessages(comm,NULL,sizes,&nrecvs));
      PetscCall(PetscGatherMessageLengths(comm,nsends,nrecvs,sizes,&onodes1,&olengths1));
      PetscCall(PetscSortMPIIntWithArray(nrecvs,onodes1,olengths1));
      recvtotal = 0; for (i=0; i<nrecvs; i++) recvtotal += olengths1[i];

      /* post receives:  rvalues - rows I will own; count - nu */
      PetscCall(PetscMalloc3(recvtotal,&rvalues,nrecvs,&source,nrecvs,&recv_waits));
      count = 0;
      for (i=0; i<nrecvs; i++) {
        PetscCallMPI(MPI_Irecv((rvalues+count),olengths1[i],MPIU_INT,onodes1[i],tag,comm,recv_waits+i));
        count += olengths1[i];
      }

      /* do sends:
       1) starts[i] gives the starting index in svalues for stuff going to
       the ith processor
       */
      PetscCall(PetscMalloc3(cnt,&svalues,nsends,&send_waits,size,&starts));
      starts[0] = 0;
      for (i=1; i<size; i++) starts[i] = starts[i-1] + sizes[i-1];
      for (i=0; i<cnt; i++)  svalues[starts[owner[i]]++] = rows[i];
      for (i=0; i<cnt; i++)  rows[i] = rows[i] - rstart;
      red->drows = drows;
      red->dcnt  = dcnt;
      PetscCall(PetscFree(rows));

      starts[0] = 0;
      for (i=1; i<size; i++) starts[i] = starts[i-1] + sizes[i-1];
      count = 0;
      for (i=0; i<size; i++) {
        if (sizes[i]) {
          PetscCallMPI(MPI_Isend(svalues+starts[i],sizes[i],MPIU_INT,i,tag,comm,send_waits+count++));
        }
      }

      /*  wait on receives */
      count = nrecvs;
      slen  = 0;
      while (count) {
        PetscCallMPI(MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status));
        /* unpack receives into our local space */
        PetscCallMPI(MPI_Get_count(&recv_status,MPIU_INT,&n));
        slen += n;
        count--;
      }
      PetscCheck(slen == recvtotal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %" PetscInt_FMT " not expected %" PetscInt_FMT,slen,recvtotal);
      PetscCall(ISCreateGeneral(comm,slen,rvalues,PETSC_COPY_VALUES,&red->is));

      /* free all work space */
      PetscCall(PetscFree(olengths1));
      PetscCall(PetscFree(onodes1));
      PetscCall(PetscFree3(rvalues,source,recv_waits));
      PetscCall(PetscFree2(sizes,owner));
      if (nsends) {   /* wait on sends */
        PetscCall(PetscMalloc1(nsends,&send_status));
        PetscCallMPI(MPI_Waitall(nsends,send_waits,send_status));
        PetscCall(PetscFree(send_status));
      }
      PetscCall(PetscFree3(svalues,send_waits,starts));
    } else {
      PetscCall(ISCreateGeneral(comm,cnt,rows,PETSC_OWN_POINTER,&red->is));
      red->drows = drows;
      red->dcnt  = dcnt;
      slen = cnt;
    }
    PetscCall(PetscLayoutDestroy(&map));
    PetscCall(PetscLayoutDestroy(&nmap));

    PetscCall(VecCreateMPI(comm,slen,PETSC_DETERMINE,&red->b));
    PetscCall(VecDuplicate(red->b,&red->x));
    PetscCall(MatCreateVecs(pc->pmat,&tvec,NULL));
    PetscCall(VecScatterCreate(tvec,red->is,red->b,NULL,&red->scatter));
    PetscCall(VecDestroy(&tvec));
    PetscCall(MatCreateSubMatrix(pc->pmat,red->is,red->is,MAT_INITIAL_MATRIX,&tmat));
    PetscCall(KSPSetOperators(red->ksp,tmat,tmat));
    PetscCall(MatDestroy(&tmat));
  }

  /* get diagonal portion of matrix */
  PetscCall(PetscFree(red->diag));
  PetscCall(PetscMalloc1(red->dcnt,&red->diag));
  PetscCall(MatCreateVecs(pc->pmat,&diag,NULL));
  PetscCall(MatGetDiagonal(pc->pmat,diag));
  PetscCall(VecGetArrayRead(diag,&d));
  for (i=0; i<red->dcnt; i++) red->diag[i] = 1.0/d[red->drows[i]];
  PetscCall(VecRestoreArrayRead(diag,&d));
  PetscCall(VecDestroy(&diag));
  PetscCall(KSPSetUp(red->ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Redistribute(PC pc,Vec b,Vec x)
{
  PC_Redistribute   *red = (PC_Redistribute*)pc->data;
  PetscInt          dcnt   = red->dcnt,i;
  const PetscInt    *drows = red->drows;
  PetscScalar       *xwork;
  const PetscScalar *bwork,*diag = red->diag;

  PetscFunctionBegin;
  if (!red->work) {
    PetscCall(VecDuplicate(b,&red->work));
  }
  /* compute the rows of solution that have diagonal entries only */
  PetscCall(VecSet(x,0.0));         /* x = diag(A)^{-1} b */
  PetscCall(VecGetArray(x,&xwork));
  PetscCall(VecGetArrayRead(b,&bwork));
  for (i=0; i<dcnt; i++) xwork[drows[i]] = diag[i]*bwork[drows[i]];
  PetscCall(PetscLogFlops(dcnt));
  PetscCall(VecRestoreArray(red->work,&xwork));
  PetscCall(VecRestoreArrayRead(b,&bwork));
  /* update the right hand side for the reduced system with diagonal rows (and corresponding columns) removed */
  PetscCall(MatMult(pc->pmat,x,red->work));
  PetscCall(VecAYPX(red->work,-1.0,b));   /* red->work = b - A x */

  PetscCall(VecScatterBegin(red->scatter,red->work,red->b,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(red->scatter,red->work,red->b,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(KSPSolve(red->ksp,red->b,red->x));
  PetscCall(KSPCheckSolve(red->ksp,pc,red->x));
  PetscCall(VecScatterBegin(red->scatter,red->x,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(red->scatter,red->x,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Redistribute(PC pc)
{
  PC_Redistribute *red = (PC_Redistribute*)pc->data;

  PetscFunctionBegin;
  PetscCall(VecScatterDestroy(&red->scatter));
  PetscCall(ISDestroy(&red->is));
  PetscCall(VecDestroy(&red->b));
  PetscCall(VecDestroy(&red->x));
  PetscCall(KSPDestroy(&red->ksp));
  PetscCall(VecDestroy(&red->work));
  PetscCall(PetscFree(red->drows));
  PetscCall(PetscFree(red->diag));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Redistribute(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Redistribute *red = (PC_Redistribute*)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPSetFromOptions(red->ksp));
  PetscFunctionReturn(0);
}

/*@
   PCRedistributeGetKSP - Gets the KSP created by the PCREDISTRIBUTE

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  innerksp - the inner KSP

   Level: advanced

@*/
PetscErrorCode  PCRedistributeGetKSP(PC pc,KSP *innerksp)
{
  PC_Redistribute *red = (PC_Redistribute*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(innerksp,2);
  *innerksp = red->ksp;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
     PCREDISTRIBUTE - Redistributes a matrix for load balancing, removing the rows (and the corresponding columns) that only have a diagonal entry and then
     applys a KSP to that new smaller matrix

     Options for the redistribute preconditioners can be set with -redistribute_ksp_xxx <values> and -redistribute_pc_xxx <values>

     Notes:
    Usually run this with -ksp_type preonly

     If you have used `MatZeroRows()` to eliminate (for example, Dirichlet) boundary conditions for a symmetric problem then you can use, for example, -ksp_type preonly
     -pc_type redistribute -redistribute_ksp_type cg -redistribute_pc_type bjacobi -redistribute_sub_pc_type icc to take advantage of the symmetry.

     This does NOT call a partitioner to reorder rows to lower communication; the ordering of the rows in the original matrix and redistributed matrix is the same.

     Developer Notes:
    Should add an option to this preconditioner to use a partitioner to redistribute the rows to lower communication.

   Level: intermediate

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PCRedistributeGetKSP()`, `MatZeroRows()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Redistribute(PC pc)
{
  PC_Redistribute *red;
  const char      *prefix;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&red));
  pc->data = (void*)red;

  pc->ops->apply          = PCApply_Redistribute;
  pc->ops->applytranspose = NULL;
  pc->ops->setup          = PCSetUp_Redistribute;
  pc->ops->destroy        = PCDestroy_Redistribute;
  pc->ops->setfromoptions = PCSetFromOptions_Redistribute;
  pc->ops->view           = PCView_Redistribute;

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc),&red->ksp));
  PetscCall(KSPSetErrorIfNotConverged(red->ksp,pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)red->ksp,(PetscObject)pc,1));
  PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)red->ksp));
  PetscCall(PCGetOptionsPrefix(pc,&prefix));
  PetscCall(KSPSetOptionsPrefix(red->ksp,prefix));
  PetscCall(KSPAppendOptionsPrefix(red->ksp,"redistribute_"));
  PetscFunctionReturn(0);
}
