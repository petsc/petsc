
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  if (iascii) {
    CHKERRMPI(MPIU_Allreduce(&red->dcnt,&ncnt,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
    CHKERRQ(MatGetSize(pc->pmat,&N,NULL));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"    Number rows eliminated %D Percentage rows eliminated %g\n",ncnt,100.0*((PetscReal)ncnt)/((PetscReal)N)));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Redistribute preconditioner: \n"));
    CHKERRQ(KSPView(red->ksp,viewer));
  } else if (isstring) {
    CHKERRQ(PetscViewerStringSPrintf(viewer," Redistribute preconditioner"));
    CHKERRQ(KSPView(red->ksp,viewer));
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
    CHKERRQ(KSPGetOperators(red->ksp,NULL,&tmat));
    CHKERRQ(MatCreateSubMatrix(pc->pmat,red->is,red->is,MAT_REUSE_MATRIX,&tmat));
    CHKERRQ(KSPSetOperators(red->ksp,tmat,tmat));
  } else {
    PetscInt NN;

    CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
    CHKERRMPI(MPI_Comm_size(comm,&size));
    CHKERRQ(PetscObjectGetNewTag((PetscObject)pc,&tag));

    /* count non-diagonal rows on process */
    CHKERRQ(MatGetOwnershipRange(pc->mat,&rstart,&rend));
    cnt  = 0;
    for (i=rstart; i<rend; i++) {
      CHKERRQ(MatGetRow(pc->mat,i,&nz,&cols,&values));
      for (PetscInt j=0; j<nz; j++) {
        if (values[j] != 0 && cols[j] != i) {
          cnt++;
          break;
        }
      }
      CHKERRQ(MatRestoreRow(pc->mat,i,&nz,&cols,&values));
    }
    CHKERRQ(PetscMalloc1(cnt,&rows));
    CHKERRQ(PetscMalloc1(rend - rstart - cnt,&drows));

    /* list non-diagonal rows on process */
    cnt = 0; dcnt = 0;
    for (i=rstart; i<rend; i++) {
      PetscBool diagonly = PETSC_TRUE;
      CHKERRQ(MatGetRow(pc->mat,i,&nz,&cols,&values));
      for (PetscInt j=0; j<nz; j++) {
        if (values[j] != 0 && cols[j] != i) {
          diagonly = PETSC_FALSE;
          break;
        }
      }
      if (!diagonly) rows[cnt++] = i;
      else drows[dcnt++] = i - rstart;
      CHKERRQ(MatRestoreRow(pc->mat,i,&nz,&cols,&values));
    }

    /* create PetscLayout for non-diagonal rows on each process */
    CHKERRQ(PetscLayoutCreate(comm,&map));
    CHKERRQ(PetscLayoutSetLocalSize(map,cnt));
    CHKERRQ(PetscLayoutSetBlockSize(map,1));
    CHKERRQ(PetscLayoutSetUp(map));
    rstart = map->rstart;
    rend   = map->rend;

    /* create PetscLayout for load-balanced non-diagonal rows on each process */
    CHKERRQ(PetscLayoutCreate(comm,&nmap));
    CHKERRMPI(MPIU_Allreduce(&cnt,&ncnt,1,MPIU_INT,MPI_SUM,comm));
    CHKERRQ(PetscLayoutSetSize(nmap,ncnt));
    CHKERRQ(PetscLayoutSetBlockSize(nmap,1));
    CHKERRQ(PetscLayoutSetUp(nmap));

    CHKERRQ(MatGetSize(pc->pmat,&NN,NULL));
    CHKERRQ(PetscInfo(pc,"Number of diagonal rows eliminated %d, percentage eliminated %g\n",NN-ncnt,((PetscReal)(NN-ncnt))/((PetscReal)(NN))));

    if (size > 1) {
      /* the following block of code assumes MPI can send messages to self, which is not supported for MPI-uni hence we need to handle the size 1 case as a special case */
      /*
       this code is taken from VecScatterCreate_PtoS()
       Determines what rows need to be moved where to
       load balance the non-diagonal rows
       */
      /*  count number of contributors to each processor */
      CHKERRQ(PetscMalloc2(size,&sizes,cnt,&owner));
      CHKERRQ(PetscArrayzero(sizes,size));
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
      CHKERRQ(PetscGatherNumberOfMessages(comm,NULL,sizes,&nrecvs));
      CHKERRQ(PetscGatherMessageLengths(comm,nsends,nrecvs,sizes,&onodes1,&olengths1));
      CHKERRQ(PetscSortMPIIntWithArray(nrecvs,onodes1,olengths1));
      recvtotal = 0; for (i=0; i<nrecvs; i++) recvtotal += olengths1[i];

      /* post receives:  rvalues - rows I will own; count - nu */
      CHKERRQ(PetscMalloc3(recvtotal,&rvalues,nrecvs,&source,nrecvs,&recv_waits));
      count = 0;
      for (i=0; i<nrecvs; i++) {
        CHKERRMPI(MPI_Irecv((rvalues+count),olengths1[i],MPIU_INT,onodes1[i],tag,comm,recv_waits+i));
        count += olengths1[i];
      }

      /* do sends:
       1) starts[i] gives the starting index in svalues for stuff going to
       the ith processor
       */
      CHKERRQ(PetscMalloc3(cnt,&svalues,nsends,&send_waits,size,&starts));
      starts[0] = 0;
      for (i=1; i<size; i++) starts[i] = starts[i-1] + sizes[i-1];
      for (i=0; i<cnt; i++)  svalues[starts[owner[i]]++] = rows[i];
      for (i=0; i<cnt; i++)  rows[i] = rows[i] - rstart;
      red->drows = drows;
      red->dcnt  = dcnt;
      CHKERRQ(PetscFree(rows));

      starts[0] = 0;
      for (i=1; i<size; i++) starts[i] = starts[i-1] + sizes[i-1];
      count = 0;
      for (i=0; i<size; i++) {
        if (sizes[i]) {
          CHKERRMPI(MPI_Isend(svalues+starts[i],sizes[i],MPIU_INT,i,tag,comm,send_waits+count++));
        }
      }

      /*  wait on receives */
      count = nrecvs;
      slen  = 0;
      while (count) {
        CHKERRMPI(MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status));
        /* unpack receives into our local space */
        CHKERRMPI(MPI_Get_count(&recv_status,MPIU_INT,&n));
        slen += n;
        count--;
      }
      PetscCheckFalse(slen != recvtotal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not expected %D",slen,recvtotal);
      CHKERRQ(ISCreateGeneral(comm,slen,rvalues,PETSC_COPY_VALUES,&red->is));

      /* free all work space */
      CHKERRQ(PetscFree(olengths1));
      CHKERRQ(PetscFree(onodes1));
      CHKERRQ(PetscFree3(rvalues,source,recv_waits));
      CHKERRQ(PetscFree2(sizes,owner));
      if (nsends) {   /* wait on sends */
        CHKERRQ(PetscMalloc1(nsends,&send_status));
        CHKERRMPI(MPI_Waitall(nsends,send_waits,send_status));
        CHKERRQ(PetscFree(send_status));
      }
      CHKERRQ(PetscFree3(svalues,send_waits,starts));
    } else {
      CHKERRQ(ISCreateGeneral(comm,cnt,rows,PETSC_OWN_POINTER,&red->is));
      red->drows = drows;
      red->dcnt  = dcnt;
      slen = cnt;
    }
    CHKERRQ(PetscLayoutDestroy(&map));
    CHKERRQ(PetscLayoutDestroy(&nmap));

    CHKERRQ(VecCreateMPI(comm,slen,PETSC_DETERMINE,&red->b));
    CHKERRQ(VecDuplicate(red->b,&red->x));
    CHKERRQ(MatCreateVecs(pc->pmat,&tvec,NULL));
    CHKERRQ(VecScatterCreate(tvec,red->is,red->b,NULL,&red->scatter));
    CHKERRQ(VecDestroy(&tvec));
    CHKERRQ(MatCreateSubMatrix(pc->pmat,red->is,red->is,MAT_INITIAL_MATRIX,&tmat));
    CHKERRQ(KSPSetOperators(red->ksp,tmat,tmat));
    CHKERRQ(MatDestroy(&tmat));
  }

  /* get diagonal portion of matrix */
  CHKERRQ(PetscFree(red->diag));
  CHKERRQ(PetscMalloc1(red->dcnt,&red->diag));
  CHKERRQ(MatCreateVecs(pc->pmat,&diag,NULL));
  CHKERRQ(MatGetDiagonal(pc->pmat,diag));
  CHKERRQ(VecGetArrayRead(diag,&d));
  for (i=0; i<red->dcnt; i++) red->diag[i] = 1.0/d[red->drows[i]];
  CHKERRQ(VecRestoreArrayRead(diag,&d));
  CHKERRQ(VecDestroy(&diag));
  CHKERRQ(KSPSetUp(red->ksp));
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
    CHKERRQ(VecDuplicate(b,&red->work));
  }
  /* compute the rows of solution that have diagonal entries only */
  CHKERRQ(VecSet(x,0.0));         /* x = diag(A)^{-1} b */
  CHKERRQ(VecGetArray(x,&xwork));
  CHKERRQ(VecGetArrayRead(b,&bwork));
  for (i=0; i<dcnt; i++) xwork[drows[i]] = diag[i]*bwork[drows[i]];
  CHKERRQ(PetscLogFlops(dcnt));
  CHKERRQ(VecRestoreArray(red->work,&xwork));
  CHKERRQ(VecRestoreArrayRead(b,&bwork));
  /* update the right hand side for the reduced system with diagonal rows (and corresponding columns) removed */
  CHKERRQ(MatMult(pc->pmat,x,red->work));
  CHKERRQ(VecAYPX(red->work,-1.0,b));   /* red->work = b - A x */

  CHKERRQ(VecScatterBegin(red->scatter,red->work,red->b,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(red->scatter,red->work,red->b,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(KSPSolve(red->ksp,red->b,red->x));
  CHKERRQ(KSPCheckSolve(red->ksp,pc,red->x));
  CHKERRQ(VecScatterBegin(red->scatter,red->x,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(red->scatter,red->x,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Redistribute(PC pc)
{
  PC_Redistribute *red = (PC_Redistribute*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(VecScatterDestroy(&red->scatter));
  CHKERRQ(ISDestroy(&red->is));
  CHKERRQ(VecDestroy(&red->b));
  CHKERRQ(VecDestroy(&red->x));
  CHKERRQ(KSPDestroy(&red->ksp));
  CHKERRQ(VecDestroy(&red->work));
  CHKERRQ(PetscFree(red->drows));
  CHKERRQ(PetscFree(red->diag));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Redistribute(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Redistribute *red = (PC_Redistribute*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(KSPSetFromOptions(red->ksp));
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
     PCREDISTRIBUTE - Redistributes a matrix for load balancing, removing the rows that only have a diagonal entry and then applys a KSP to that new matrix

     Options for the redistribute preconditioners can be set with -redistribute_ksp_xxx <values> and -redistribute_pc_xxx <values>

     Notes:
    Usually run this with -ksp_type preonly

     If you have used MatZeroRows() to eliminate (for example, Dirichlet) boundary conditions for a symmetric problem then you can use, for example, -ksp_type preonly
     -pc_type redistribute -redistribute_ksp_type cg -redistribute_pc_type bjacobi -redistribute_sub_pc_type icc to take advantage of the symmetry.

     This does NOT call a partitioner to reorder rows to lower communication; the ordering of the rows in the original matrix and redistributed matrix is the same.

     Developer Notes:
    Should add an option to this preconditioner to use a partitioner to redistribute the rows to lower communication.

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCRedistributeGetKSP()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Redistribute(PC pc)
{
  PC_Redistribute *red;
  const char      *prefix;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&red));
  pc->data = (void*)red;

  pc->ops->apply          = PCApply_Redistribute;
  pc->ops->applytranspose = NULL;
  pc->ops->setup          = PCSetUp_Redistribute;
  pc->ops->destroy        = PCDestroy_Redistribute;
  pc->ops->setfromoptions = PCSetFromOptions_Redistribute;
  pc->ops->view           = PCView_Redistribute;

  CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)pc),&red->ksp));
  CHKERRQ(KSPSetErrorIfNotConverged(red->ksp,pc->erroriffailure));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)red->ksp,(PetscObject)pc,1));
  CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)red->ksp));
  CHKERRQ(PCGetOptionsPrefix(pc,&prefix));
  CHKERRQ(KSPSetOptionsPrefix(red->ksp,prefix));
  CHKERRQ(KSPAppendOptionsPrefix(red->ksp,"redistribute_"));
  PetscFunctionReturn(0);
}
