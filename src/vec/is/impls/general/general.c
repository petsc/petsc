
/*
     Provides the functions for index sets (IS) defined by a list of integers.
*/
#include <../src/vec/is/impls/general/general.h> /*I  "petscis.h"  I*/
#include <petscvec.h>

#undef __FUNCT__  
#define __FUNCT__ "ISDuplicate_General" 
PetscErrorCode ISDuplicate_General(IS is,IS *newIS)
{
  PetscErrorCode ierr;
  IS_General     *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  ierr = ISCreateGeneral(((PetscObject)is)->comm,sub->n,sub->idx,PETSC_COPY_VALUES,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISDestroy_General" 
PetscErrorCode ISDestroy_General(IS is)
{
  IS_General     *is_general = (IS_General*)is->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is_general->allocated) {
    ierr = PetscFree(is_general->idx);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISGeneralSetIndices_C","",0);CHKERRQ(ierr);
  ierr = PetscFree(is->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISIdentity_General" 
PetscErrorCode ISIdentity_General(IS is,PetscBool  *ident)
{
  IS_General *is_general = (IS_General*)is->data;
  PetscInt   i,n = is_general->n,*idx = is_general->idx;

  PetscFunctionBegin;
  is->isidentity = PETSC_TRUE;
  *ident         = PETSC_TRUE;
  for (i=0; i<n; i++) {
    if (idx[i] != i) {
      is->isidentity = PETSC_FALSE;
      *ident         = PETSC_FALSE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISCopy_General"
static PetscErrorCode ISCopy_General(IS is,IS isy)
{
  IS_General *is_general = (IS_General*)is->data,*isy_general = (IS_General*)isy->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is_general->n != isy_general->n || is_general->N != isy_general->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Index sets incompatible");
  isy_general->sorted = is_general->sorted;
  ierr = PetscMemcpy(isy_general->idx,is_general->idx,is_general->n*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISOnComm_General"
PetscErrorCode ISOnComm_General(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  PetscErrorCode ierr;
  IS_General     *sub = (IS_General*)is->data;

  PetscFunctionBegin;
  if (mode == PETSC_OWN_POINTER) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Cannot use PETSC_OWN_POINTER");
  ierr = ISCreateGeneral(comm,sub->n,sub->idx,mode,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSetBlockSize_General"
static PetscErrorCode ISSetBlockSize_General(IS is,PetscInt bs)
{
  IS_General *sub = (IS_General*)is->data;

  PetscFunctionBegin;
  if (sub->N % bs) SETERRQ2(((PetscObject)is)->comm,PETSC_ERR_ARG_SIZ,"Block size %D does not divide global size %D",bs,sub->N);
  if (sub->n % bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Block size %D does not divide local size %D",bs,sub->n);
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt i,j;
    for (i=0; i<sub->n; i+=bs) {
      for (j=0; j<bs; j++) {
        if (sub->idx[i+j] != sub->idx[i]+j) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Index set does not have block structure, cannot set block size to %D",bs);
      }
    }
  }
#endif
  is->bs = bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISContiguousLocal_General"
static PetscErrorCode ISContiguousLocal_General(IS is,PetscInt gstart,PetscInt gend,PetscInt *start,PetscBool *contig)
{
  IS_General *sub = (IS_General*)is->data;
  PetscInt   i,p;

  PetscFunctionBegin;
  *start  = 0;
  *contig = PETSC_TRUE;
  if (!sub->n) PetscFunctionReturn(0);
  p = sub->idx[0];
  if (p < gstart) goto nomatch;
  *start = p - gstart;
  if (sub->n > gend-p) goto nomatch;
  for (i=1; i<sub->n; i++,p++) {
    if (sub->idx[i] != p+1) goto nomatch;
  }
  PetscFunctionReturn(0);
nomatch:
  *start = -1;
  *contig = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetIndices_General" 
PetscErrorCode ISGetIndices_General(IS in,const PetscInt *idx[])
{
  IS_General *sub = (IS_General*)in->data;

  PetscFunctionBegin;
  *idx = sub->idx; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISRestoreIndices_General" 
PetscErrorCode ISRestoreIndices_General(IS in,const PetscInt *idx[])
{
  IS_General *sub = (IS_General*)in->data;

  PetscFunctionBegin;
  if (*idx != sub->idx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from ISGetIndices()");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetSize_General" 
PetscErrorCode ISGetSize_General(IS is,PetscInt *size)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *size = sub->N; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetLocalSize_General" 
PetscErrorCode ISGetLocalSize_General(IS is,PetscInt *size)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *size = sub->n; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISInvertPermutation_General" 
PetscErrorCode ISInvertPermutation_General(IS is,PetscInt nlocal,IS *isout)
{
  IS_General     *sub = (IS_General *)is->data;
  PetscInt       i,*ii,n = sub->n,nstart;
  const PetscInt *idx = sub->idx;
  PetscMPIInt    size;
  IS             istmp,nistmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)is)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscMalloc(n*sizeof(PetscInt),&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ii[idx[i]] = i;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_OWN_POINTER,isout);CHKERRQ(ierr);
    ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
  } else {
    /* crude, nonscalable get entire IS on each processor */
    if (nlocal == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Do not yet support nlocal of PETSC_DECIDE");
    ierr = ISAllGather(is,&istmp);CHKERRQ(ierr);
    ierr = ISSetPermutation(istmp);CHKERRQ(ierr);
    ierr = ISInvertPermutation(istmp,PETSC_DECIDE,&nistmp);CHKERRQ(ierr);
    ierr = ISDestroy(&istmp);CHKERRQ(ierr);
    /* get the part we need */
    ierr    = MPI_Scan(&nlocal,&nstart,1,MPIU_INT,MPI_SUM,((PetscObject)is)->comm);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    {
      PetscMPIInt rank;
      ierr = MPI_Comm_rank(((PetscObject)is)->comm,&rank);CHKERRQ(ierr);
      if (rank == size-1) {
        if (nstart != sub->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sum of nlocal lengths %d != total IS length %d",nstart,sub->N);
      }
    }
#endif
    nstart -= nlocal;
    ierr    = ISGetIndices(nistmp,&idx);CHKERRQ(ierr);
    ierr    = ISCreateGeneral(((PetscObject)is)->comm,nlocal,idx+nstart,PETSC_COPY_VALUES,isout);CHKERRQ(ierr);    
    ierr    = ISRestoreIndices(nistmp,&idx);CHKERRQ(ierr);
    ierr    = ISDestroy(&nistmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISView_General_Binary"
PetscErrorCode ISView_General_Binary(IS is,PetscViewer viewer)
{
  PetscErrorCode ierr;
  IS_General     *isa = (IS_General*) is->data;
  PetscMPIInt    rank,size,mesgsize,tag = ((PetscObject)viewer)->tag, mesglen;
  PetscInt       len,j,tr[2];
  int            fdes;
  MPI_Status     status;
  PetscInt       message_count,flowcontrolcount,*values;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fdes);CHKERRQ(ierr);

  /* determine maximum message to arrive */
  ierr = MPI_Comm_rank(((PetscObject)is)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)is)->comm,&size);CHKERRQ(ierr);

  tr[0] = IS_FILE_CLASSID;
  tr[1] = isa->N;
  ierr = PetscViewerBinaryWrite(viewer,tr,2,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MPI_Reduce(&isa->n,&len,1,MPIU_INT,MPI_SUM,0,((PetscObject)is)->comm);CHKERRQ(ierr);

  ierr = PetscViewerFlowControlStart(viewer,&message_count,&flowcontrolcount);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinaryWrite(fdes,isa->idx,isa->n,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);

    ierr = PetscMalloc(len*sizeof(PetscInt),&values);CHKERRQ(ierr);
    mesgsize = PetscMPIIntCast(len);
    /* receive and save messages */
    for (j=1; j<size; j++) {
      ierr = PetscViewerFlowControlStepMaster(viewer,j,message_count,flowcontrolcount);CHKERRQ(ierr);
      ierr = MPI_Recv(values,mesgsize,MPIU_INT,j,tag,((PetscObject)is)->comm,&status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status,MPIU_INT,&mesglen);CHKERRQ(ierr);         
      ierr = PetscBinaryWrite(fdes,values,(PetscInt)mesglen,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlowControlEndMaster(viewer,message_count);CHKERRQ(ierr);
    ierr = PetscFree(values);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerFlowControlStepWorker(viewer,rank,message_count);CHKERRQ(ierr);
    mesgsize = PetscMPIIntCast(isa->n);
    ierr = MPI_Send(isa->idx,mesgsize,MPIU_INT,0,tag,((PetscObject)is)->comm);CHKERRQ(ierr);
    ierr = PetscViewerFlowControlEndWorker(viewer,message_count);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISView_General" 
PetscErrorCode ISView_General(IS is,PetscViewer viewer)
{
  IS_General     *sub = (IS_General *)is->data;
  PetscErrorCode ierr;
  PetscInt       i,n = sub->n,*idx = sub->idx;
  PetscBool      iascii,isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) {
    MPI_Comm    comm;
    PetscMPIInt rank,size;

    ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);      
    if (size > 1) {
      if (is->isperm) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Index set is permutation\n",rank);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of indices in set %D\n",rank,n);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D %D\n",rank,i,idx[i]);CHKERRQ(ierr);
      }
    } else {
      if (is->isperm) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Index set is permutation\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Number of indices in set %D\n",n);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%D %D\n",i,idx[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);      
  } else if (isbinary) {
    ierr = ISView_General_Binary(is,viewer);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSort_General" 
PetscErrorCode ISSort_General(IS is)
{
  IS_General     *sub = (IS_General *)is->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sub->sorted) PetscFunctionReturn(0);
  ierr = PetscSortInt(sub->n,sub->idx);CHKERRQ(ierr);
  sub->sorted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSorted_General" 
PetscErrorCode ISSorted_General(IS is,PetscBool  *flg)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *flg = sub->sorted;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISToGeneral_General" 
PetscErrorCode  ISToGeneral_General(IS is)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static struct _ISOps myops = { ISGetSize_General,
                               ISGetLocalSize_General,
                               ISGetIndices_General,
                               ISRestoreIndices_General,
                               ISInvertPermutation_General,
                               ISSort_General,
                               ISSorted_General,
                               ISDuplicate_General,
                               ISDestroy_General,
                               ISView_General,
                               ISIdentity_General,
                               ISCopy_General,
                               ISToGeneral_General,
                               ISOnComm_General,
                               ISSetBlockSize_General,
                               ISContiguousLocal_General
};

#undef __FUNCT__  
#define __FUNCT__ "ISCreateGeneral_Private" 
PetscErrorCode ISCreateGeneral_Private(IS is)
{
  PetscErrorCode ierr;
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       n = sub->n,i,min,max;
  const PetscInt *idx = sub->idx;
  PetscBool      sorted = PETSC_TRUE;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MPI_Allreduce(&n,&sub->N,1,MPIU_INT,MPI_SUM,((PetscObject)is)->comm);CHKERRQ(ierr);
  for (i=1; i<n; i++) {
    if (idx[i] < idx[i-1]) {sorted = PETSC_FALSE; break;}
  }
  if (n) {min = max = idx[0];} else {min = max = 0;}
  for (i=1; i<n; i++) {
    if (idx[i] < min) min = idx[i];
    if (idx[i] > max) max = idx[i];
  }
  sub->sorted = sorted;
  is->min     = min;
  is->max     = max;
  is->isperm     = PETSC_FALSE;
  is->isidentity = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-is_view",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(((PetscObject)is)->comm,&viewer);CHKERRQ(ierr);
    ierr = ISView(is,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISCreateGeneral" 
/*@
   ISCreateGeneral - Creates a data structure for an index set 
   containing a list of integers.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator
.  n - the length of the index set
.  idx - the list of integers
-  mode - see PetscCopyMode for meaning of this flag.

   Output Parameter:
.  is - the new index set

   Notes:
   When the communicator is not MPI_COMM_SELF, the operations on IS are NOT
   conceptually the same as MPI_Group operations. The IS are then
   distributed sets of indices and thus certain operations on them are
   collective.

   
   Level: beginner

  Concepts: index sets^creating
  Concepts: IS^creating

.seealso: ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISCreateGeneral(MPI_Comm comm,PetscInt n,const PetscInt idx[],PetscCopyMode mode,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISCreate(comm,is);CHKERRQ(ierr);
  ierr = ISSetType(*is,ISGENERAL);CHKERRQ(ierr);
  ierr = ISGeneralSetIndices(*is,n,idx,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGeneralSetIndices" 
/*@
   ISGeneralSetIndices - Sets the indices for an ISGENERAL index set

   Collective on IS

   Input Parameters:
+  is - the index set
.  n - the length of the index set
.  idx - the list of integers
-  mode - see PetscCopyMode for meaning of this flag.

   Level: beginner

  Concepts: index sets^creating
  Concepts: IS^creating

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISGeneralSetIndices(IS is,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISGeneralSetIndices_C",(IS,PetscInt,const PetscInt[],PetscCopyMode),(is,n,idx,mode));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "ISGeneralSetIndices_General" 
PetscErrorCode  ISGeneralSetIndices_General(IS is,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscErrorCode ierr;
  IS_General     *sub = (IS_General*)is->data;

  PetscFunctionBegin;
  if (n < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) {PetscValidIntPointer(idx,3);}

  if (sub->allocated) {ierr = PetscFree(sub->idx);CHKERRQ(ierr);}
  sub->n = n;
  if (mode == PETSC_COPY_VALUES) {
    ierr           = PetscMalloc(n*sizeof(PetscInt),&sub->idx);CHKERRQ(ierr);
    ierr           = PetscLogObjectMemory(is,n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr           = PetscMemcpy(sub->idx,idx,n*sizeof(PetscInt));CHKERRQ(ierr);
    sub->allocated = PETSC_TRUE;
  } else if (mode == PETSC_OWN_POINTER) {
    sub->idx       = (PetscInt*)idx;
    sub->allocated = PETSC_TRUE;
  } else {
    sub->idx       = (PetscInt*)idx;
    sub->allocated = PETSC_FALSE;
  }    
  ierr = ISCreateGeneral_Private(is); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END




EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "ISCreate_General" 
PetscErrorCode  ISCreate_General(IS is)
{
  PetscErrorCode ierr;
  IS_General     *sub;

  PetscFunctionBegin;
  ierr = PetscMemcpy(is->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ierr = PetscNewLog(is,IS_General,&sub);CHKERRQ(ierr);
  is->data = (void*)sub;
  is->bs   = 1;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISGeneralSetIndices_C","ISGeneralSetIndices_General",ISGeneralSetIndices_General);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END






