
/*
  This file defines a "solve the problem redistributely on each subgroup of processor" preconditioner.
*/
#include <petsc-private/pcimpl.h>     /*I "petscksp.h" I*/
#include <petscksp.h>

typedef struct {
  KSP          ksp;
  Vec          x,b;
  VecScatter   scatter;
  IS           is;
  PetscInt     dcnt,*drows;   /* these are the local rows that have only diagonal entry */
  PetscScalar  *diag;
  Vec          work;
} PC_Redistribute;

#undef __FUNCT__  
#define __FUNCT__ "PCView_Redistribute"
static PetscErrorCode PCView_Redistribute(PC pc,PetscViewer viewer)
{
  PC_Redistribute *red = (PC_Redistribute*)pc->data;
  PetscErrorCode  ierr;
  PetscBool       iascii,isstring;
  PetscInt        ncnt,N;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = MPI_Allreduce(&red->dcnt,&ncnt,1,MPIU_INT,MPI_SUM,((PetscObject)pc)->comm);CHKERRQ(ierr);
    ierr = MatGetSize(pc->pmat,&N,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"    Number rows eliminated %D Percentage rows eliminated %g\n",ncnt,100.0*((PetscReal)ncnt)/((PetscReal)N));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Redistribute preconditioner: \n");CHKERRQ(ierr);
    ierr = KSPView(red->ksp,viewer);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," Redistribute preconditioner");CHKERRQ(ierr);
    ierr = KSPView(red->ksp,viewer);CHKERRQ(ierr);
  } else SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PC redistribute",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Redistribute"
static PetscErrorCode PCSetUp_Redistribute(PC pc)
{
  PC_Redistribute   *red = (PC_Redistribute*)pc->data;
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  PetscInt          rstart,rend,i,nz,cnt,*rows,ncnt,dcnt,*drows;
  PetscLayout       map,nmap;
  PetscMPIInt       size,imdex,tag,n;
  PetscInt          *source = PETSC_NULL;
  PetscMPIInt       *nprocs = PETSC_NULL,nrecvs;
  PetscInt          j,nsends;
  PetscInt          *owner = PETSC_NULL,*starts = PETSC_NULL,count,slen;
  PetscInt          *rvalues,*svalues,recvtotal;
  PetscMPIInt       *onodes1,*olengths1;
  MPI_Request       *send_waits = PETSC_NULL,*recv_waits = PETSC_NULL;
  MPI_Status        recv_status,*send_status;
  Vec               tvec,diag;
  Mat               tmat;
  const PetscScalar *d;

  PetscFunctionBegin;
  if (pc->setupcalled) {
    ierr = KSPGetOperators(red->ksp,PETSC_NULL,&tmat,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(pc->pmat,red->is,red->is,MAT_REUSE_MATRIX,&tmat);CHKERRQ(ierr);
    ierr = KSPSetOperators(red->ksp,tmat,tmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  } else {
    PetscInt NN;

    ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = PetscObjectGetNewTag((PetscObject)pc,&tag);CHKERRQ(ierr);

    /* count non-diagonal rows on process */
    ierr = MatGetOwnershipRange(pc->mat,&rstart,&rend);CHKERRQ(ierr);
    cnt  = 0;  
    for (i=rstart; i<rend; i++) {
      ierr = MatGetRow(pc->mat,i,&nz,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      if (nz > 1) cnt++;
      ierr = MatRestoreRow(pc->mat,i,&nz,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(cnt*sizeof(PetscInt),&rows);CHKERRQ(ierr);
    ierr = PetscMalloc((rend - rstart - cnt)*sizeof(PetscInt),&drows);CHKERRQ(ierr);

    /* list non-diagonal rows on process */
    cnt  = 0; dcnt = 0;  
    for (i=rstart; i<rend; i++) {
      ierr = MatGetRow(pc->mat,i,&nz,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      if (nz > 1) rows[cnt++] = i;
      else drows[dcnt++] = i - rstart;
      ierr = MatRestoreRow(pc->mat,i,&nz,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }

    /* create PetscLayout for non-diagonal rows on each process */
    ierr = PetscLayoutCreate(comm,&map);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(map,cnt);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(map,1);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
    rstart = map->rstart;
    rend   = map->rend;
    
    /* create PetscLayout for load-balanced non-diagonal rows on each process */
    ierr = PetscLayoutCreate(comm,&nmap);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&cnt,&ncnt,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(nmap,ncnt);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(nmap,1);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(nmap);CHKERRQ(ierr);

    ierr = MatGetSize(pc->pmat,&NN,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscInfo2(pc,"Number of diagonal rows eliminated %d, percentage eliminated %g\n",NN-ncnt,((PetscReal)(NN-ncnt))/((PetscReal)(NN)));
    /*  
	this code is taken from VecScatterCreate_PtoS() 
	Determines what rows need to be moved where to 
	load balance the non-diagonal rows 
    */
    /*  count number of contributors to each processor */
    ierr = PetscMalloc2(size,PetscMPIInt,&nprocs,cnt,PetscInt,&owner);CHKERRQ(ierr);
    ierr = PetscMemzero(nprocs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
    j      = 0;
    nsends = 0;
    for (i=rstart; i<rend; i++) {
      if (i < nmap->range[j]) j = 0;
      for (; j<size; j++) {
	if (i < nmap->range[j+1]) {
	  if (!nprocs[j]++) nsends++;
	  owner[i-rstart] = j; 
	  break;
	}
      }
    }
    /* inform other processors of number of messages and max length*/
    ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,nprocs,&nrecvs);CHKERRQ(ierr);
    ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,nprocs,&onodes1,&olengths1);CHKERRQ(ierr);
    ierr = PetscSortMPIIntWithArray(nrecvs,onodes1,olengths1);CHKERRQ(ierr);
    recvtotal = 0; for (i=0; i<nrecvs; i++) recvtotal += olengths1[i];
    
    /* post receives:  rvalues - rows I will own; count - nu */
    ierr = PetscMalloc3(recvtotal,PetscInt,&rvalues,nrecvs,PetscInt,&source,nrecvs,MPI_Request,&recv_waits);CHKERRQ(ierr);
    count  = 0;
    for (i=0; i<nrecvs; i++) {
      ierr  = MPI_Irecv((rvalues+count),olengths1[i],MPIU_INT,onodes1[i],tag,comm,recv_waits+i);CHKERRQ(ierr);
      count += olengths1[i];
    }

    /* do sends:
       1) starts[i] gives the starting index in svalues for stuff going to 
       the ith processor
    */
    ierr = PetscMalloc3(cnt,PetscInt,&svalues,nsends,MPI_Request,&send_waits,size,PetscInt,&starts);CHKERRQ(ierr);
    starts[0]  = 0; 
    for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
    for (i=0; i<cnt; i++) {
      svalues[starts[owner[i]]++] = rows[i];
    }
    for (i=0; i<cnt; i++) rows[i] = rows[i] - rstart;
    red->drows = drows;
    red->dcnt  = dcnt;
    ierr = PetscFree(rows);CHKERRQ(ierr);

    starts[0] = 0;
    for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
    count = 0;
    for (i=0; i<size; i++) {
      if (nprocs[i]) {
	ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPIU_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
      }
    }
    
    /*  wait on receives */
    count  = nrecvs; 
    slen   = 0;
    while (count) {
      ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
      /* unpack receives into our local space */
      ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
      slen += n;
      count--;
    }
    if (slen != recvtotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not expected %D",slen,recvtotal);
    
    ierr = ISCreateGeneral(comm,slen,rvalues,PETSC_COPY_VALUES,&red->is);CHKERRQ(ierr);
    
    /* free up all work space */
    ierr = PetscFree(olengths1);CHKERRQ(ierr);
    ierr = PetscFree(onodes1);CHKERRQ(ierr);
    ierr = PetscFree3(rvalues,source,recv_waits);CHKERRQ(ierr);
    ierr = PetscFree2(nprocs,owner);CHKERRQ(ierr);
    if (nsends) {   /* wait on sends */
      ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
      ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
      ierr = PetscFree(send_status);CHKERRQ(ierr);
    }
    ierr = PetscFree3(svalues,send_waits,starts);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&nmap);CHKERRQ(ierr);

    ierr = VecCreateMPI(comm,slen,PETSC_DETERMINE,&red->b);CHKERRQ(ierr);
    ierr = VecDuplicate(red->b,&red->x);CHKERRQ(ierr);
    ierr = MatGetVecs(pc->pmat,&tvec,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecScatterCreate(tvec,red->is,red->b,PETSC_NULL,&red->scatter);CHKERRQ(ierr);
    ierr = VecDestroy(&tvec);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(pc->pmat,red->is,red->is,MAT_INITIAL_MATRIX,&tmat);CHKERRQ(ierr);
    ierr = KSPSetOperators(red->ksp,tmat,tmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(&tmat);CHKERRQ(ierr);
  }

  /* get diagonal portion of matrix */
  ierr = PetscMalloc(red->dcnt*sizeof(PetscScalar),&red->diag);CHKERRQ(ierr);
  ierr = MatGetVecs(pc->pmat,&diag,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetDiagonal(pc->pmat,diag);CHKERRQ(ierr);
  ierr = VecGetArrayRead(diag,&d);CHKERRQ(ierr);
  for (i=0; i<red->dcnt; i++) {
    red->diag[i] = 1.0/d[red->drows[i]];
  }
  ierr = VecRestoreArrayRead(diag,&d);CHKERRQ(ierr);
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  ierr = KSPSetUp(red->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Redistribute"
static PetscErrorCode PCApply_Redistribute(PC pc,Vec b,Vec x)
{
  PC_Redistribute   *red = (PC_Redistribute*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          dcnt = red->dcnt,i;
  const PetscInt    *drows = red->drows;
  PetscScalar       *xwork;
  const PetscScalar *bwork,*diag = red->diag;

  PetscFunctionBegin;
  if (!red->work) {
    ierr = VecDuplicate(b,&red->work);CHKERRQ(ierr);
  }
  /* compute the rows of solution that have diagonal entries only */
  ierr = VecSet(x,0.0);CHKERRQ(ierr);         /* x = diag(A)^{-1} b */
  ierr = VecGetArray(x,&xwork);CHKERRQ(ierr);
  ierr = VecGetArrayRead(b,&bwork);CHKERRQ(ierr);
  for (i=0; i<dcnt; i++) {
    xwork[drows[i]] = diag[i]*bwork[drows[i]];
  }
  ierr = PetscLogFlops(dcnt);CHKERRQ(ierr);
  ierr = VecRestoreArray(red->work,&xwork);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b,&bwork);CHKERRQ(ierr);
  /* update the right hand side for the reduced system with diagonal rows (and corresponding columns) removed */
  ierr = MatMult(pc->pmat,x,red->work);CHKERRQ(ierr);
  ierr = VecAYPX(red->work,-1.0,b);CHKERRQ(ierr);   /* red->work = b - A x */

  ierr = VecScatterBegin(red->scatter,red->work,red->b,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->scatter,red->work,red->b,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = KSPSolve(red->ksp,red->b,red->x);CHKERRQ(ierr);
  ierr = VecScatterBegin(red->scatter,red->x,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->scatter,red->x,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Redistribute"
static PetscErrorCode PCDestroy_Redistribute(PC pc)
{
  PC_Redistribute *red = (PC_Redistribute*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecScatterDestroy(&red->scatter);CHKERRQ(ierr);
  ierr = ISDestroy(&red->is);CHKERRQ(ierr);
  ierr = VecDestroy(&red->b);CHKERRQ(ierr);
  ierr = VecDestroy(&red->x);CHKERRQ(ierr);
  ierr = KSPDestroy(&red->ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&red->work);CHKERRQ(ierr);
  ierr = PetscFree(red->drows);
  ierr = PetscFree(red->diag);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Redistribute"
static PetscErrorCode PCSetFromOptions_Redistribute(PC pc)
{
  PetscErrorCode  ierr;
  PC_Redistribute *red = (PC_Redistribute*)pc->data;

  PetscFunctionBegin;
  ierr = KSPSetFromOptions(red->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCRedistributeGetKSP"
/*@
   PCRedistributeGetKSP - Gets the KSP created by the PCREDISTRIBUTE

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  innerksp - the inner KSP

   Level: advanced

.keywords: PC, redistribute solve
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

     Notes:  Usually run this with -ksp_type preonly

     If you have used MatZeroRows() to eliminate (for example, Dirichlet) boundary conditions for a symmetric problem then you can use, for example, -ksp_type preonly 
     -pc_type redistribute -redistribute_ksp_type cg -redistribute_pc_type bjacobi -redistribute_sub_pc_type icc to take advantage of the symmetry.

     This does NOT call a partitioner to reorder rows to lower communication; the ordering of the rows in the original matrix and redistributed matrix is the same.

     Developer Notes: Should add an option to this preconditioner to use a partitioner to redistribute the rows to lower communication.

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCRedistributeGetKSP()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Redistribute"
PetscErrorCode  PCCreate_Redistribute(PC pc)
{
  PetscErrorCode  ierr;
  PC_Redistribute *red;
  const char      *prefix;
  
  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_Redistribute,&red);CHKERRQ(ierr);
  pc->data            = (void*)red; 

  pc->ops->apply           = PCApply_Redistribute;
  pc->ops->applytranspose  = 0;
  pc->ops->setup           = PCSetUp_Redistribute;
  pc->ops->destroy         = PCDestroy_Redistribute;
  pc->ops->setfromoptions  = PCSetFromOptions_Redistribute;
  pc->ops->view            = PCView_Redistribute;    

  ierr = KSPCreate(((PetscObject)pc)->comm,&red->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)red->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(pc,red->ksp);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(red->ksp,prefix);CHKERRQ(ierr); 
  ierr = KSPAppendOptionsPrefix(red->ksp,"redistribute_");CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
EXTERN_C_END
