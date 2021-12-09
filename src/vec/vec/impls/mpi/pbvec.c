
/*
   This file contains routines for Parallel vector operations.
 */
#include <petscsys.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/

PetscErrorCode VecDot_MPI(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_Seq(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPI(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_Seq(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

extern PetscErrorCode VecView_MPI_Draw(Vec,PetscViewer);

static PetscErrorCode VecPlaceArray_MPI(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;
  Vec_MPI        *v = (Vec_MPI*)vin->data;

  PetscFunctionBegin;
  if (v->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
  v->unplacedarray = v->array;  /* save previous array so reset can bring it back */
  v->array         = (PetscScalar*)a;
  if (v->localrep) {
    ierr = VecPlaceArray(v->localrep,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_MPI(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)win),v);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*v)->map);CHKERRQ(ierr);

  ierr = VecCreate_MPI_Private(*v,PETSC_TRUE,w->nghost,NULL);CHKERRQ(ierr);
  vw   = (Vec_MPI*)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,PetscAbs(win->map->bs),win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
    ierr = VecRestoreArray(*v,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)*v,(PetscObject)vw->localrep);CHKERRQ(ierr);

    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      ierr = PetscObjectReference((PetscObject)vw->localupdate);CHKERRQ(ierr);
    }
  }

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash   = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;

  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);

  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetOption_MPI(Vec V,VecOption op,PetscBool flag)
{
  Vec_MPI        *v = (Vec_MPI*)V->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case VEC_IGNORE_OFF_PROC_ENTRIES: V->stash.donotstash = flag;
    break;
  case VEC_IGNORE_NEGATIVE_INDICES: V->stash.ignorenegidx = flag;
    break;
  case VEC_SUBSET_OFF_PROC_ENTRIES:
    v->assembly_subset = flag; /* See the same logic in MatAssembly wrt MAT_SUBSET_OFF_PROC_ENTRIES */
    if (!v->assembly_subset) { /* User indicates "do not reuse the communication pattern" */
      ierr = VecAssemblyReset_MPI(V);CHKERRQ(ierr); /* Reset existing pattern to free memory */
      v->first_assembly_done = PETSC_FALSE; /* Mark the first assembly is not done */
    }
    break;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode VecResetArray_MPI(Vec vin)
{
  Vec_MPI        *v = (Vec_MPI*)vin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = NULL;
  if (v->localrep) {
    ierr = VecResetArray(v->localrep);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblySend_MPI_Private(MPI_Comm comm,const PetscMPIInt tag[],PetscMPIInt rankid,PetscMPIInt rank,void *sdata,MPI_Request req[],void *ctx)
{
  Vec X = (Vec)ctx;
  Vec_MPI *x = (Vec_MPI*)X->data;
  VecAssemblyHeader *hdr = (VecAssemblyHeader*)sdata;
  PetscInt bs = X->map->bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* x->first_assembly_done indicates we are reusing a communication network. In that case, some
     messages can be empty, but we have to send them this time if we sent them before because the
     receiver is expecting them.
   */
  if (hdr->count || (x->first_assembly_done && x->sendptrs[rankid].ints)) {
    ierr = MPI_Isend(x->sendptrs[rankid].ints,hdr->count,MPIU_INT,rank,tag[0],comm,&req[0]);CHKERRMPI(ierr);
    ierr = MPI_Isend(x->sendptrs[rankid].scalars,hdr->count,MPIU_SCALAR,rank,tag[1],comm,&req[1]);CHKERRMPI(ierr);
  }
  if (hdr->bcount || (x->first_assembly_done && x->sendptrs[rankid].intb)) {
    ierr = MPI_Isend(x->sendptrs[rankid].intb,hdr->bcount,MPIU_INT,rank,tag[2],comm,&req[2]);CHKERRMPI(ierr);
    ierr = MPI_Isend(x->sendptrs[rankid].scalarb,hdr->bcount*bs,MPIU_SCALAR,rank,tag[3],comm,&req[3]);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblyRecv_MPI_Private(MPI_Comm comm,const PetscMPIInt tag[],PetscMPIInt rank,void *rdata,MPI_Request req[],void *ctx)
{
  Vec X = (Vec)ctx;
  Vec_MPI *x = (Vec_MPI*)X->data;
  VecAssemblyHeader *hdr = (VecAssemblyHeader*)rdata;
  PetscErrorCode ierr;
  PetscInt bs = X->map->bs;
  VecAssemblyFrame *frame;

  PetscFunctionBegin;
  ierr = PetscSegBufferGet(x->segrecvframe,1,&frame);CHKERRQ(ierr);

  if (hdr->count) {
    ierr = PetscSegBufferGet(x->segrecvint,hdr->count,&frame->ints);CHKERRQ(ierr);
    ierr = MPI_Irecv(frame->ints,hdr->count,MPIU_INT,rank,tag[0],comm,&req[0]);CHKERRMPI(ierr);
    ierr = PetscSegBufferGet(x->segrecvscalar,hdr->count,&frame->scalars);CHKERRQ(ierr);
    ierr = MPI_Irecv(frame->scalars,hdr->count,MPIU_SCALAR,rank,tag[1],comm,&req[1]);CHKERRMPI(ierr);
    frame->pendings = 2;
  } else {
    frame->ints = NULL;
    frame->scalars = NULL;
    frame->pendings = 0;
  }

  if (hdr->bcount) {
    ierr = PetscSegBufferGet(x->segrecvint,hdr->bcount,&frame->intb);CHKERRQ(ierr);
    ierr = MPI_Irecv(frame->intb,hdr->bcount,MPIU_INT,rank,tag[2],comm,&req[2]);CHKERRMPI(ierr);
    ierr = PetscSegBufferGet(x->segrecvscalar,hdr->bcount*bs,&frame->scalarb);CHKERRQ(ierr);
    ierr = MPI_Irecv(frame->scalarb,hdr->bcount*bs,MPIU_SCALAR,rank,tag[3],comm,&req[3]);CHKERRMPI(ierr);
    frame->pendingb = 2;
  } else {
    frame->intb = NULL;
    frame->scalarb = NULL;
    frame->pendingb = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblyBegin_MPI_BTS(Vec X)
{
  Vec_MPI        *x = (Vec_MPI*)X->data;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt       i,j,jb,bs;

  PetscFunctionBegin;
  if (X->stash.donotstash) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)X,&comm);CHKERRQ(ierr);
  ierr = VecGetBlockSize(X,&bs);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    InsertMode addv;
    ierr = MPIU_Allreduce((PetscEnum*)&X->stash.insertmode,(PetscEnum*)&addv,1,MPIU_ENUM,MPI_BOR,comm);CHKERRMPI(ierr);
    if (addv == (ADD_VALUES|INSERT_VALUES)) SETERRQ(comm,PETSC_ERR_ARG_NOTSAMETYPE,"Some processors inserted values while others added");
  }
  X->bstash.insertmode = X->stash.insertmode; /* Block stash implicitly tracks InsertMode of scalar stash */

  ierr = VecStashSortCompress_Private(&X->stash);CHKERRQ(ierr);
  ierr = VecStashSortCompress_Private(&X->bstash);CHKERRQ(ierr);

  if (!x->sendranks) {
    PetscMPIInt nowners,bnowners,*owners,*bowners;
    PetscInt ntmp;
    ierr = VecStashGetOwnerList_Private(&X->stash,X->map,&nowners,&owners);CHKERRQ(ierr);
    ierr = VecStashGetOwnerList_Private(&X->bstash,X->map,&bnowners,&bowners);CHKERRQ(ierr);
    ierr = PetscMergeMPIIntArray(nowners,owners,bnowners,bowners,&ntmp,&x->sendranks);CHKERRQ(ierr);
    x->nsendranks = ntmp;
    ierr = PetscFree(owners);CHKERRQ(ierr);
    ierr = PetscFree(bowners);CHKERRQ(ierr);
    ierr = PetscMalloc1(x->nsendranks,&x->sendhdr);CHKERRQ(ierr);
    ierr = PetscCalloc1(x->nsendranks,&x->sendptrs);CHKERRQ(ierr);
  }
  for (i=0,j=0,jb=0; i<x->nsendranks; i++) {
    PetscMPIInt rank = x->sendranks[i];
    x->sendhdr[i].insertmode = X->stash.insertmode;
    /* Initialize pointers for non-empty stashes the first time around.  Subsequent assemblies with
     * VEC_SUBSET_OFF_PROC_ENTRIES will leave the old pointers (dangling because the stash has been collected) when
     * there is nothing new to send, so that size-zero messages get sent instead. */
    x->sendhdr[i].count = 0;
    if (X->stash.n) {
      x->sendptrs[i].ints    = &X->stash.idx[j];
      x->sendptrs[i].scalars = &X->stash.array[j];
      for (; j<X->stash.n && X->stash.idx[j] < X->map->range[rank+1]; j++) x->sendhdr[i].count++;
    }
    x->sendhdr[i].bcount = 0;
    if (X->bstash.n) {
      x->sendptrs[i].intb    = &X->bstash.idx[jb];
      x->sendptrs[i].scalarb = &X->bstash.array[jb*bs];
      for (; jb<X->bstash.n && X->bstash.idx[jb]*bs < X->map->range[rank+1]; jb++) x->sendhdr[i].bcount++;
    }
  }

  if (!x->segrecvint) {ierr = PetscSegBufferCreate(sizeof(PetscInt),1000,&x->segrecvint);CHKERRQ(ierr);}
  if (!x->segrecvscalar) {ierr = PetscSegBufferCreate(sizeof(PetscScalar),1000,&x->segrecvscalar);CHKERRQ(ierr);}
  if (!x->segrecvframe) {ierr = PetscSegBufferCreate(sizeof(VecAssemblyFrame),50,&x->segrecvframe);CHKERRQ(ierr);}
  if (x->first_assembly_done) { /* this is not the first assembly */
    PetscMPIInt tag[4];
    for (i=0; i<4; i++) {ierr = PetscCommGetNewTag(comm,&tag[i]);CHKERRQ(ierr);}
    for (i=0; i<x->nsendranks; i++) {
      ierr = VecAssemblySend_MPI_Private(comm,tag,i,x->sendranks[i],x->sendhdr+i,x->sendreqs+4*i,X);CHKERRQ(ierr);
    }
    for (i=0; i<x->nrecvranks; i++) {
      ierr = VecAssemblyRecv_MPI_Private(comm,tag,x->recvranks[i],x->recvhdr+i,x->recvreqs+4*i,X);CHKERRQ(ierr);
    }
    x->use_status = PETSC_TRUE;
  } else { /* First time assembly */
    ierr = PetscCommBuildTwoSidedFReq(comm,3,MPIU_INT,x->nsendranks,x->sendranks,(PetscInt*)x->sendhdr,&x->nrecvranks,&x->recvranks,&x->recvhdr,4,&x->sendreqs,&x->recvreqs,VecAssemblySend_MPI_Private,VecAssemblyRecv_MPI_Private,X);CHKERRQ(ierr);
    x->use_status = PETSC_FALSE;
  }

  /* The first_assembly_done flag is only meaningful when x->assembly_subset is set.
     This line says when assembly_subset is set, then we mark that the first assembly is done.
   */
  x->first_assembly_done = x->assembly_subset;

  {
    PetscInt nstash,reallocs;
    ierr = VecStashGetInfo_Private(&X->stash,&nstash,&reallocs);CHKERRQ(ierr);
    ierr = PetscInfo2(X,"Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
    ierr = VecStashGetInfo_Private(&X->bstash,&nstash,&reallocs);CHKERRQ(ierr);
    ierr = PetscInfo2(X,"Block-Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblyEnd_MPI_BTS(Vec X)
{
  Vec_MPI *x = (Vec_MPI*)X->data;
  PetscInt bs = X->map->bs;
  PetscMPIInt npending,*some_indices,r;
  MPI_Status  *some_statuses;
  PetscScalar *xarray;
  PetscErrorCode ierr;
  VecAssemblyFrame *frame;

  PetscFunctionBegin;
  if (X->stash.donotstash) {
    X->stash.insertmode = NOT_SET_VALUES;
    X->bstash.insertmode = NOT_SET_VALUES;
    PetscFunctionReturn(0);
  }

  if (!x->segrecvframe) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing segrecvframe! Probably you forgot to call VecAssemblyBegin first");
  ierr = VecGetArray(X,&xarray);CHKERRQ(ierr);
  ierr = PetscSegBufferExtractInPlace(x->segrecvframe,&frame);CHKERRQ(ierr);
  ierr = PetscMalloc2(4*x->nrecvranks,&some_indices,x->use_status?4*x->nrecvranks:0,&some_statuses);CHKERRQ(ierr);
  for (r=0,npending=0; r<x->nrecvranks; r++) npending += frame[r].pendings + frame[r].pendingb;
  while (npending>0) {
    PetscMPIInt ndone=0,ii;
    /* Filling MPI_Status fields requires some resources from the MPI library.  We skip it on the first assembly, or
     * when VEC_SUBSET_OFF_PROC_ENTRIES has not been set, because we could exchange exact sizes in the initial
     * rendezvous.  When the rendezvous is elided, however, we use MPI_Status to get actual message lengths, so that
     * subsequent assembly can set a proper subset of the values. */
    ierr = MPI_Waitsome(4*x->nrecvranks,x->recvreqs,&ndone,some_indices,x->use_status?some_statuses:MPI_STATUSES_IGNORE);CHKERRMPI(ierr);
    for (ii=0; ii<ndone; ii++) {
      PetscInt i = some_indices[ii]/4,j,k;
      InsertMode imode = (InsertMode)x->recvhdr[i].insertmode;
      PetscInt *recvint;
      PetscScalar *recvscalar;
      PetscBool intmsg = (PetscBool)(some_indices[ii]%2 == 0);
      PetscBool blockmsg = (PetscBool)((some_indices[ii]%4)/2 == 1);
      npending--;
      if (!blockmsg) { /* Scalar stash */
        PetscMPIInt count;
        if (--frame[i].pendings > 0) continue;
        if (x->use_status) {
          ierr = MPI_Get_count(&some_statuses[ii],intmsg ? MPIU_INT : MPIU_SCALAR,&count);CHKERRMPI(ierr);
        } else count = x->recvhdr[i].count;
        for (j=0,recvint=frame[i].ints,recvscalar=frame[i].scalars; j<count; j++,recvint++) {
          PetscInt loc = *recvint - X->map->rstart;
          if (*recvint < X->map->rstart || X->map->rend <= *recvint) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Received vector entry %" PetscInt_FMT " out of local range [%" PetscInt_FMT ",%" PetscInt_FMT ")]",*recvint,X->map->rstart,X->map->rend);
          switch (imode) {
          case ADD_VALUES:
            xarray[loc] += *recvscalar++;
            break;
          case INSERT_VALUES:
            xarray[loc] = *recvscalar++;
            break;
          default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Insert mode not supported 0x%x",imode);
          }
        }
      } else {                  /* Block stash */
        PetscMPIInt count;
        if (--frame[i].pendingb > 0) continue;
        if (x->use_status) {
          ierr = MPI_Get_count(&some_statuses[ii],intmsg ? MPIU_INT : MPIU_SCALAR,&count);CHKERRMPI(ierr);
          if (!intmsg) count /= bs; /* Convert from number of scalars to number of blocks */
        } else count = x->recvhdr[i].bcount;
        for (j=0,recvint=frame[i].intb,recvscalar=frame[i].scalarb; j<count; j++,recvint++) {
          PetscInt loc = (*recvint)*bs - X->map->rstart;
          switch (imode) {
          case ADD_VALUES:
            for (k=loc; k<loc+bs; k++) xarray[k] += *recvscalar++;
            break;
          case INSERT_VALUES:
            for (k=loc; k<loc+bs; k++) xarray[k] = *recvscalar++;
            break;
          default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Insert mode not supported 0x%x",imode);
          }
        }
      }
    }
  }
  ierr = VecRestoreArray(X,&xarray);CHKERRQ(ierr);
  ierr = MPI_Waitall(4*x->nsendranks,x->sendreqs,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);
  ierr = PetscFree2(some_indices,some_statuses);CHKERRQ(ierr);
  if (x->assembly_subset) {
    void *dummy;                /* reset segbuffers */
    ierr = PetscSegBufferExtractInPlace(x->segrecvint,&dummy);CHKERRQ(ierr);
    ierr = PetscSegBufferExtractInPlace(x->segrecvscalar,&dummy);CHKERRQ(ierr);
  } else {
    ierr = VecAssemblyReset_MPI(X);CHKERRQ(ierr);
  }

  X->stash.insertmode = NOT_SET_VALUES;
  X->bstash.insertmode = NOT_SET_VALUES;
  ierr = VecStashScatterEnd_Private(&X->stash);CHKERRQ(ierr);
  ierr = VecStashScatterEnd_Private(&X->bstash);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAssemblyReset_MPI(Vec X)
{
  Vec_MPI *x = (Vec_MPI*)X->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(x->sendreqs);CHKERRQ(ierr);
  ierr = PetscFree(x->recvreqs);CHKERRQ(ierr);
  ierr = PetscFree(x->sendranks);CHKERRQ(ierr);
  ierr = PetscFree(x->recvranks);CHKERRQ(ierr);
  ierr = PetscFree(x->sendhdr);CHKERRQ(ierr);
  ierr = PetscFree(x->recvhdr);CHKERRQ(ierr);
  ierr = PetscFree(x->sendptrs);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&x->segrecvint);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&x->segrecvscalar);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&x->segrecvframe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetFromOptions_MPI(PetscOptionItems *PetscOptionsObject,Vec X)
{
#if !defined(PETSC_HAVE_MPIUNI)
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE,set;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"VecMPI Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-vec_assembly_legacy","Use MPI 1 version of assembly","",flg,&flg,&set);CHKERRQ(ierr);
  if (set) {
    X->ops->assemblybegin = flg ? VecAssemblyBegin_MPI : VecAssemblyBegin_MPI_BTS;
    X->ops->assemblyend   = flg ? VecAssemblyEnd_MPI   : VecAssemblyEnd_MPI_BTS;
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
#else
  PetscFunctionBegin;
  X->ops->assemblybegin = VecAssemblyBegin_MPI;
  X->ops->assemblyend   = VecAssemblyEnd_MPI;
#endif
  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = { VecDuplicate_MPI, /* 1 */
                                VecDuplicateVecs_Default,
                                VecDestroyVecs_Default,
                                VecDot_MPI,
                                VecMDot_MPI,
                                VecNorm_MPI,
                                VecTDot_MPI,
                                VecMTDot_MPI,
                                VecScale_Seq,
                                VecCopy_Seq, /* 10 */
                                VecSet_Seq,
                                VecSwap_Seq,
                                VecAXPY_Seq,
                                VecAXPBY_Seq,
                                VecMAXPY_Seq,
                                VecAYPX_Seq,
                                VecWAXPY_Seq,
                                VecAXPBYPCZ_Seq,
                                VecPointwiseMult_Seq,
                                VecPointwiseDivide_Seq,
                                VecSetValues_MPI, /* 20 */
                                VecAssemblyBegin_MPI_BTS,
                                VecAssemblyEnd_MPI_BTS,
                                NULL,
                                VecGetSize_MPI,
                                VecGetSize_Seq,
                                NULL,
                                VecMax_MPI,
                                VecMin_MPI,
                                VecSetRandom_Seq,
                                VecSetOption_MPI,
                                VecSetValuesBlocked_MPI,
                                VecDestroy_MPI,
                                VecView_MPI,
                                VecPlaceArray_MPI,
                                VecReplaceArray_Seq,
                                VecDot_Seq,
                                VecTDot_Seq,
                                VecNorm_Seq,
                                VecMDot_Seq,
                                VecMTDot_Seq,
                                VecLoad_Default,
                                VecReciprocal_Default,
                                VecConjugate_Seq,
                                NULL,
                                NULL,
                                VecResetArray_MPI,
                                VecSetFromOptions_MPI,/*set from options */
                                VecMaxPointwiseDivide_Seq,
                                VecPointwiseMax_Seq,
                                VecPointwiseMaxAbs_Seq,
                                VecPointwiseMin_Seq,
                                VecGetValues_MPI,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                VecStrideGather_Default,
                                VecStrideScatter_Default,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                VecStrideSubSetGather_Default,
                                VecStrideSubSetScatter_Default,
                                NULL,
                                NULL,
                                NULL
};

/*
    VecCreate_MPI_Private - Basic create routine called by VecCreate_MPI() (i.e. VecCreateMPI()),
    VecCreateMPIWithArray(), VecCreate_Shared() (i.e. VecCreateShared()), VecCreateGhost(),
    VecDuplicate_MPI(), VecCreateGhostWithArray(), VecDuplicate_MPI(), and VecDuplicate_Shared()

    If alloc is true and array is NULL then this routine allocates the space, otherwise
    no space is allocated.
*/
PetscErrorCode VecCreate_MPI_Private(Vec v,PetscBool alloc,PetscInt nghost,const PetscScalar array[])
{
  Vec_MPI        *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr           = PetscNewLog(v,&s);CHKERRQ(ierr);
  v->data        = (void*)s;
  ierr           = PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  s->nghost      = nghost;
  v->petscnative = PETSC_TRUE;
  if (array) v->offloadmask = PETSC_OFFLOAD_CPU;

  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);

  s->array           = (PetscScalar*)array;
  s->array_allocated = NULL;
  if (alloc && !array) {
    PetscInt n = v->map->n+nghost;
    ierr               = PetscCalloc1(n,&s->array);CHKERRQ(ierr);
    ierr               = PetscLogObjectMemory((PetscObject)v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array_allocated = s->array;
  }

  /* By default parallel vectors do not have local representation */
  s->localrep    = NULL;
  s->localupdate = NULL;

  v->stash.insertmode = NOT_SET_VALUES;
  v->bstash.insertmode = NOT_SET_VALUES;
  /* create the stashes. The block-size for bstash is set later when
     VecSetValuesBlocked is called.
  */
  ierr = VecStashCreate_Private(PetscObjectComm((PetscObject)v),1,&v->stash);CHKERRQ(ierr);
  ierr = VecStashCreate_Private(PetscObjectComm((PetscObject)v),PetscAbs(v->map->bs),&v->bstash);CHKERRQ(ierr);

#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscMatlabEnginePut_C",VecMatlabEnginePut_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscMatlabEngineGet_C",VecMatlabEngineGet_Default);CHKERRQ(ierr);
#endif
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECMPI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   VECMPI - VECMPI = "mpi" - The basic parallel vector

   Options Database Keys:
. -vec_type mpi - sets the vector type to VECMPI during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateMPI()
M*/

PetscErrorCode VecCreate_MPI(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_TRUE,0,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   VECSTANDARD = "standard" - A VECSEQ on one process and VECMPI on more than one process

   Options Database Keys:
. -vec_type standard - sets a vector type to standard on calls to VecSetFromOptions()

  Level: beginner

.seealso: VecCreateSeq(), VecCreateMPI()
M*/

PETSC_EXTERN PetscErrorCode VecCreate_Standard(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v),&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQ);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPI);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPIWithArray - Creates a parallel, array-style vector,
   where the user provides the array space to store the vector values.

   Collective

   Input Parameters:
+  comm  - the MPI communicator to use
.  bs    - block size, same meaning as VecSetBlockSize()
.  n     - local vector length, cannot be PETSC_DECIDE
.  N     - global vector length (or PETSC_DECIDE to have calculated)
-  array - the user provided array to store the vector values

   Output Parameter:
.  vv - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/
PetscErrorCode  VecCreateMPIWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_FALSE,0,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCreateGhostWithArray - Creates a parallel vector with ghost padding on each processor;
   the caller allocates the array space.

   Collective

   Input Parameters:
+  comm - the MPI communicator to use
.  n - local vector length
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost points
.  ghosts - global indices of ghost points (or NULL if not needed), these do not need to be in increasing order (sorted)
-  array - the space to store the vector values (as long as n + nghost)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   This also automatically sets the ISLocalToGlobalMapping() for this vector.

   Level: advanced

.seealso: VecCreate(), VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhost(), VecCreateSeqWithArray(), VecCreateMPIWithArray(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray(), VecMPISetGhost()

@*/
PetscErrorCode  VecCreateGhostWithArray(MPI_Comm comm,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],const PetscScalar array[],Vec *vv)
{
  PetscErrorCode         ierr;
  Vec_MPI                *w;
  PetscScalar            *larray;
  IS                     from,to;
  ISLocalToGlobalMapping ltog;
  PetscInt               rstart,i,*indices;

  PetscFunctionBegin;
  *vv = NULL;

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  /* Create global representation */
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_TRUE,nghost,array);CHKERRQ(ierr);
  w    = (Vec_MPI*)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArray(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,n+nghost,larray,&w->localrep);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)*vv,(PetscObject)w->localrep);CHKERRQ(ierr);
  ierr = VecRestoreArray(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values
  */
  ierr = ISCreateGeneral(comm,nghost,ghosts,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,nghost,n,1,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)*vv,(PetscObject)w->localupdate);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

  /* set local to global mapping for ghosted vector */
  ierr = PetscMalloc1(n+nghost,&indices);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(*vv,&rstart,NULL);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    indices[i] = rstart + i;
  }
  for (i=0; i<nghost; i++) {
    indices[n+i] = ghosts[i];
  }
  ierr = ISLocalToGlobalMappingCreate(comm,1,n+nghost,indices,PETSC_OWN_POINTER,&ltog);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*vv,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecCreateGhost - Creates a parallel vector with ghost padding on each processor.

   Collective

   Input Parameters:
+  comm - the MPI communicator to use
.  n - local vector length
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost points
-  ghosts - global indices of ghost points, these do not need to be in increasing order (sorted)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   This also automatically sets the ISLocalToGlobalMapping() for this vector.

   Level: advanced

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), VecGhostUpdateBegin(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecGhostUpdateEnd(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray(), VecMPISetGhost()

@*/
PetscErrorCode  VecCreateGhost(MPI_Comm comm,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateGhostWithArray(comm,n,N,nghost,ghosts,NULL,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecMPISetGhost - Sets the ghost points for an MPI ghost vector

   Collective on Vec

   Input Parameters:
+  vv - the MPI vector
.  nghost - number of local ghost points
-  ghosts - global indices of ghost points, these do not need to be in increasing order (sorted)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   This also automatically sets the ISLocalToGlobalMapping() for this vector.

   You must call this AFTER you have set the type of the vector (with VecSetType()) and the size (with VecSetSizes()).

   Level: advanced

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), VecGhostUpdateBegin(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecGhostUpdateEnd(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray()

@*/
PetscErrorCode  VecMPISetGhost(Vec vv,PetscInt nghost,const PetscInt ghosts[])
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)vv,VECMPI,&flg);CHKERRQ(ierr);
  /* if already fully existant VECMPI then basically destroy it and rebuild with ghosting */
  if (flg) {
    PetscInt               n,N;
    Vec_MPI                *w;
    PetscScalar            *larray;
    IS                     from,to;
    ISLocalToGlobalMapping ltog;
    PetscInt               rstart,i,*indices;
    MPI_Comm               comm;

    ierr = PetscObjectGetComm((PetscObject)vv,&comm);CHKERRQ(ierr);
    n    = vv->map->n;
    N    = vv->map->N;
    ierr = (*vv->ops->destroy)(vv);CHKERRQ(ierr);
    ierr = VecSetSizes(vv,n,N);CHKERRQ(ierr);
    ierr = VecCreate_MPI_Private(vv,PETSC_TRUE,nghost,NULL);CHKERRQ(ierr);
    w    = (Vec_MPI*)(vv)->data;
    /* Create local representation */
    ierr = VecGetArray(vv,&larray);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,n+nghost,larray,&w->localrep);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)vv,(PetscObject)w->localrep);CHKERRQ(ierr);
    ierr = VecRestoreArray(vv,&larray);CHKERRQ(ierr);

    /*
     Create scatter context for scattering (updating) ghost values
     */
    ierr = ISCreateGeneral(comm,nghost,ghosts,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,nghost,n,1,&to);CHKERRQ(ierr);
    ierr = VecScatterCreate(vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)vv,(PetscObject)w->localupdate);CHKERRQ(ierr);
    ierr = ISDestroy(&to);CHKERRQ(ierr);
    ierr = ISDestroy(&from);CHKERRQ(ierr);

    /* set local to global mapping for ghosted vector */
    ierr = PetscMalloc1(n+nghost,&indices);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(vv,&rstart,NULL);CHKERRQ(ierr);

    for (i=0; i<n; i++)      indices[i]   = rstart + i;
    for (i=0; i<nghost; i++) indices[n+i] = ghosts[i];

    ierr = ISLocalToGlobalMappingCreate(comm,1,n+nghost,indices,PETSC_OWN_POINTER,&ltog);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(vv,ltog);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  } else if (vv->ops->create == VecCreate_MPI) SETERRQ(PetscObjectComm((PetscObject)vv),PETSC_ERR_ARG_WRONGSTATE,"Must set local or global size before setting ghosting");
  else if (!((PetscObject)vv)->type_name) SETERRQ(PetscObjectComm((PetscObject)vv),PETSC_ERR_ARG_WRONGSTATE,"Must set type to VECMPI before ghosting");
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------------*/
/*@C
   VecCreateGhostBlockWithArray - Creates a parallel vector with ghost padding on each processor;
   the caller allocates the array space. Indices in the ghost region are based on blocks.

   Collective

   Input Parameters:
+  comm - the MPI communicator to use
.  bs - block size
.  n - local vector length
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost blocks
.  ghosts - global indices of ghost blocks (or NULL if not needed), counts are by block not by index, these do not need to be in increasing order (sorted)
-  array - the space to store the vector values (as long as n + nghost*bs)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   n is the local vector size (total local size not the number of blocks) while nghost
   is the number of blocks in the ghost portion, i.e. the number of elements in the ghost
   portion is bs*nghost

   Level: advanced

.seealso: VecCreate(), VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhost(), VecCreateSeqWithArray(), VecCreateMPIWithArray(),
          VecCreateGhostWithArray(), VecCreateGhostBlock()

@*/
PetscErrorCode  VecCreateGhostBlockWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],const PetscScalar array[],Vec *vv)
{
  PetscErrorCode         ierr;
  Vec_MPI                *w;
  PetscScalar            *larray;
  IS                     from,to;
  ISLocalToGlobalMapping ltog;
  PetscInt               rstart,i,nb,*indices;

  PetscFunctionBegin;
  *vv = NULL;

  if (n == PETSC_DECIDE)      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size");
  if (nghost == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local ghost size");
  if (nghost < 0)             SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Ghost length must be >= 0");
  if (n % bs)                 SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size must be a multiple of block size");
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  /* Create global representation */
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  ierr = VecCreate_MPI_Private(*vv,PETSC_TRUE,nghost*bs,array);CHKERRQ(ierr);
  w    = (Vec_MPI*)(*vv)->data;
  /* Create local representation */
  ierr = VecGetArray(*vv,&larray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,n+bs*nghost,larray,&w->localrep);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)*vv,(PetscObject)w->localrep);CHKERRQ(ierr);
  ierr = VecRestoreArray(*vv,&larray);CHKERRQ(ierr);

  /*
       Create scatter context for scattering (updating) ghost values
  */
  ierr = ISCreateBlock(comm,bs,nghost,ghosts,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,bs*nghost,n,1,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(*vv,from,w->localrep,to,&w->localupdate);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)*vv,(PetscObject)w->localupdate);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

  /* set local to global mapping for ghosted vector */
  nb     = n/bs;
  ierr   = PetscMalloc1(nb+nghost,&indices);CHKERRQ(ierr);
  ierr   = VecGetOwnershipRange(*vv,&rstart,NULL);CHKERRQ(ierr);
  rstart = rstart/bs;

  for (i=0; i<nb; i++)      indices[i]    = rstart + i;
  for (i=0; i<nghost; i++)  indices[nb+i] = ghosts[i];

  ierr = ISLocalToGlobalMappingCreate(comm,bs,nb+nghost,indices,PETSC_OWN_POINTER,&ltog);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*vv,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecCreateGhostBlock - Creates a parallel vector with ghost padding on each processor.
        The indicing of the ghost points is done with blocks.

   Collective

   Input Parameters:
+  comm - the MPI communicator to use
.  bs - the block size
.  n - local vector length
.  N - global vector length (or PETSC_DECIDE to have calculated if n is given)
.  nghost - number of local ghost blocks
-  ghosts - global indices of ghost blocks, counts are by block, not by individual index, these do not need to be in increasing order (sorted)

   Output Parameter:
.  vv - the global vector representation (without ghost points as part of vector)

   Notes:
   Use VecGhostGetLocalForm() to access the local, ghosted representation
   of the vector.

   n is the local vector size (total local size not the number of blocks) while nghost
   is the number of blocks in the ghost portion, i.e. the number of elements in the ghost
   portion is bs*nghost

   Level: advanced

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecCreateGhostBlockWithArray()

@*/
PetscErrorCode  VecCreateGhostBlock(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,PetscInt nghost,const PetscInt ghosts[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateGhostBlockWithArray(comm,bs,n,N,nghost,ghosts,NULL,vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
