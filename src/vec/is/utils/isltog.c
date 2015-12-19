
#include <petsc/private/isimpl.h>    /*I "petscis.h"  I*/
#include <petscsf.h>
#include <petscviewer.h>

PetscClassId IS_LTOGM_CLASSID;
static PetscErrorCode  ISLocalToGlobalMappingGetBlockInfo_Private(ISLocalToGlobalMapping,PetscInt*,PetscInt**,PetscInt**,PetscInt***);


#undef __FUNCT__
#define __FUNCT__ "ISG2LMapApply"
PetscErrorCode ISG2LMapApply(ISLocalToGlobalMapping mapping,PetscInt n,const PetscInt in[],PetscInt out[])
{
  PetscErrorCode ierr;
  PetscInt       i,start,end;

  PetscFunctionBegin;
  if (!mapping->globals) {
    ierr = ISGlobalToLocalMappingApply(mapping,IS_GTOLM_MASK,0,0,0,0);CHKERRQ(ierr);
  }
  start = mapping->globalstart;
  end = mapping->globalend;
  for (i=0; i<n; i++) {
    if (in[i] < 0)          out[i] = in[i];
    else if (in[i] < start) out[i] = -1;
    else if (in[i] > end)   out[i] = -1;
    else                    out[i] = mapping->globals[in[i] - start];
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingGetSize"
/*@
    ISLocalToGlobalMappingGetSize - Gets the local size of a local to global mapping

    Not Collective

    Input Parameter:
.   ltog - local to global mapping

    Output Parameter:
.   n - the number of entries in the local mapping, ISLocalToGlobalMappingGetIndices() returns an array of this length

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping mapping,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  PetscValidIntPointer(n,2);
  *n = mapping->bs*mapping->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingView"
/*@C
    ISLocalToGlobalMappingView - View a local to global mapping

    Not Collective

    Input Parameters:
+   ltog - local to global mapping
-   viewer - viewer

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingView(ISLocalToGlobalMapping mapping,PetscViewer viewer)
{
  PetscInt       i;
  PetscMPIInt    rank;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mapping),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mapping),&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)mapping,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    for (i=0; i<mapping->n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D %D\n",rank,i,mapping->indices[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingCreateIS"
/*@
    ISLocalToGlobalMappingCreateIS - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not collective

    Input Parameter:
.   is - index set containing the global numbers for each local number

    Output Parameter:
.   mapping - new mapping data structure

    Notes: the block size of the IS determines the block size of the mapping
    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingCreateIS(IS is,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       n,bs;
  const PetscInt *indices;
  MPI_Comm       comm;
  PetscBool      isblock;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(mapping,2);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)is,ISBLOCK,&isblock);CHKERRQ(ierr);
  if (!isblock) {
    ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(comm,1,n,indices,PETSC_COPY_VALUES,mapping);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);
  } else {
    ierr = ISGetBlockSize(is,&bs);CHKERRQ(ierr);
    ierr = ISBlockGetIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(comm,bs,n/bs,indices,PETSC_COPY_VALUES,mapping);CHKERRQ(ierr);
    ierr = ISBlockRestoreIndices(is,&indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingCreateSF"
/*@C
    ISLocalToGlobalMappingCreateSF - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Collective

    Input Parameter:
+   sf - star forest mapping contiguous local indices to (rank, offset)
-   start - first global index on this process

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingCreateIS()
@*/
PetscErrorCode ISLocalToGlobalMappingCreateSF(PetscSF sf,PetscInt start,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       i,maxlocal,nroots,nleaves,*globals,*ltog;
  const PetscInt *ilocal;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(mapping,3);

  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,NULL);CHKERRQ(ierr);
  if (ilocal) {
    for (i=0,maxlocal=0; i<nleaves; i++) maxlocal = PetscMax(maxlocal,ilocal[i]+1);
  }
  else maxlocal = nleaves;
  ierr = PetscMalloc1(nroots,&globals);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxlocal,&ltog);CHKERRQ(ierr);
  for (i=0; i<nroots; i++) globals[i] = start + i;
  for (i=0; i<maxlocal; i++) ltog[i] = -1;
  ierr = PetscSFBcastBegin(sf,MPIU_INT,globals,ltog);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,globals,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm,1,maxlocal,ltog,PETSC_OWN_POINTER,mapping);CHKERRQ(ierr);
  ierr = PetscFree(globals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingGetBlockSize"
/*@
    ISLocalToGlobalMappingGetBlockSize - Gets the blocksize of the mapping
    ordering and a global parallel ordering.

    Not Collective

    Input Parameters:
.   mapping - mapping data structure

    Output Parameter:
.   bs - the blocksize

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockSize(ISLocalToGlobalMapping mapping,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  *bs = mapping->bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingCreate"
/*@
    ISLocalToGlobalMappingCreate - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not Collective, but communicator may have more than one process

    Input Parameters:
+   comm - MPI communicator
.   bs - the block size
.   n - the number of local elements divided by the block size, or equivalently the number of block indices
.   indices - the global index for each local element, these do not need to be in increasing order (sorted), these values should not be scaled (i.e. multiplied) by the blocksize bs
-   mode - see PetscCopyMode

    Output Parameter:
.   mapping - new mapping data structure

    Notes: There is one integer value in indices per block and it represents the actual indices bs*idx + j, where j=0,..,bs-1
    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS()
@*/
PetscErrorCode  ISLocalToGlobalMappingCreate(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscInt indices[],PetscCopyMode mode,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       *in;

  PetscFunctionBegin;
  if (n) PetscValidIntPointer(indices,3);
  PetscValidPointer(mapping,4);

  *mapping = NULL;
  ierr = ISInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(*mapping,IS_LTOGM_CLASSID,"ISLocalToGlobalMapping","Local to global mapping","IS",
                           comm,ISLocalToGlobalMappingDestroy,ISLocalToGlobalMappingView);CHKERRQ(ierr);
  (*mapping)->n             = n;
  (*mapping)->bs            = bs;
  (*mapping)->info_cached   = PETSC_FALSE;
  (*mapping)->info_free     = PETSC_FALSE;
  (*mapping)->info_procs    = NULL;
  (*mapping)->info_numprocs = NULL;
  (*mapping)->info_indices  = NULL;
  /*
    Do not create the global to local mapping. This is only created if
    ISGlobalToLocalMapping() is called
  */
  (*mapping)->globals = 0;
  if (mode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc1(n,&in);CHKERRQ(ierr);
    ierr = PetscMemcpy(in,indices,n*sizeof(PetscInt));CHKERRQ(ierr);
    (*mapping)->indices = in;
    ierr = PetscLogObjectMemory((PetscObject)*mapping,n*sizeof(PetscInt));CHKERRQ(ierr);
  } else if (mode == PETSC_OWN_POINTER) {
    (*mapping)->indices = (PetscInt*)indices;
    ierr = PetscLogObjectMemory((PetscObject)*mapping,n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  else SETERRQ(comm,PETSC_ERR_SUP,"Cannot currently use PETSC_USE_POINTER");
  ierr = PetscStrallocpy("basic",&((PetscObject)*mapping)->type_name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingDestroy"
/*@
   ISLocalToGlobalMappingDestroy - Destroys a mapping between a local (0 to n)
   ordering and a global parallel ordering.

   Note Collective

   Input Parameters:
.  mapping - mapping data structure

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*mapping) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*mapping),IS_LTOGM_CLASSID,1);
  if (--((PetscObject)(*mapping))->refct > 0) {*mapping = 0;PetscFunctionReturn(0);}
  ierr = PetscFree((*mapping)->indices);CHKERRQ(ierr);
  ierr = PetscFree((*mapping)->globals);CHKERRQ(ierr);
  ierr = PetscFree((*mapping)->info_procs);CHKERRQ(ierr);
  ierr = PetscFree((*mapping)->info_numprocs);CHKERRQ(ierr);
  if ((*mapping)->info_indices) {
    PetscInt i;

    ierr = PetscFree(((*mapping)->info_indices)[0]);CHKERRQ(ierr);
    for (i=1; i<(*mapping)->info_nproc; i++) {
      ierr = PetscFree(((*mapping)->info_indices)[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree((*mapping)->info_indices);CHKERRQ(ierr);
  }
  ierr     = PetscHeaderDestroy(mapping);CHKERRQ(ierr);
  *mapping = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingApplyIS"
/*@
    ISLocalToGlobalMappingApplyIS - Creates from an IS in the local numbering
    a new index set using the global numbering defined in an ISLocalToGlobalMapping
    context.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
-   is - index set in local numbering

    Output Parameters:
.   newis - index set in global numbering

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy(), ISGlobalToLocalMappingApply()
@*/
PetscErrorCode  ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping,IS is,IS *newis)
{
  PetscErrorCode ierr;
  PetscInt       n,*idxout;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscValidPointer(newis,3);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idxin);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&idxout);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mapping,n,idxin,idxout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&idxin);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is),n,idxout,PETSC_OWN_POINTER,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingApply"
/*@
   ISLocalToGlobalMappingApply - Takes a list of integers in a local numbering
   and converts them to the global numbering.

   Not collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local numbering

   Output Parameter:
.  out - indices in global numbering

   Notes:
   The in and out array parameters may be identical.

   Level: advanced

.seealso: ISLocalToGlobalMappingApplyBlock(), ISLocalToGlobalMappingCreate(),ISLocalToGlobalMappingDestroy(),
          ISLocalToGlobalMappingApplyIS(),AOCreateBasic(),AOApplicationToPetsc(),
          AOPetscToApplication(), ISGlobalToLocalMappingApply()

    Concepts: mapping^local to global
@*/
PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
{
  PetscInt i,bs,Nmax;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  bs   = mapping->bs;
  Nmax = bs*mapping->n;
  if (bs == 1) {
    const PetscInt *idx = mapping->indices;
    for (i=0; i<N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i],Nmax-1,i);
      out[i] = idx[in[i]];
    }
  } else {
    const PetscInt *idx = mapping->indices;
    for (i=0; i<N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i],Nmax-1,i);
      out[i] = idx[in[i]/bs]*bs + (in[i] % bs);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingApplyBlock"
/*@
   ISLocalToGlobalMappingApplyBlock - Takes a list of integers in a local block numbering  and converts them to the global block numbering

   Not collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local block numbering

   Output Parameter:
.  out - indices in global block numbering

   Notes:
   The in and out array parameters may be identical.

   Example:
     If the index values are {0,1,6,7} set with a call to ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,2,2,{0,3}) then the mapping applied to 0
     (the first block) would produce 0 and the mapping applied to 1 (the second block) would produce 3.

   Level: advanced

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),ISLocalToGlobalMappingDestroy(),
          ISLocalToGlobalMappingApplyIS(),AOCreateBasic(),AOApplicationToPetsc(),
          AOPetscToApplication(), ISGlobalToLocalMappingApply()

    Concepts: mapping^local to global
@*/
PetscErrorCode ISLocalToGlobalMappingApplyBlock(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  {
    PetscInt i,Nmax = mapping->n;
    const PetscInt *idx = mapping->indices;

    for (i=0; i<N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local block index %D too large %D (max) at %D",in[i],Nmax-1,i);
      out[i] = idx[in[i]];
    }
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "ISGlobalToLocalMappingSetUp_Private"
/*
    Creates the global fields in the ISLocalToGlobalMapping structure
*/
static PetscErrorCode ISGlobalToLocalMappingSetUp_Private(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;
  PetscInt       i,*idx = mapping->indices,n = mapping->n,end,start,*globals;

  PetscFunctionBegin;
  end   = 0;
  start = PETSC_MAX_INT;

  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    if (idx[i] < start) start = idx[i];
    if (idx[i] > end)   end   = idx[i];
  }
  if (start > end) {start = 0; end = -1;}
  mapping->globalstart = start;
  mapping->globalend   = end;

  ierr             = PetscMalloc1(end-start+2,&globals);CHKERRQ(ierr);
  mapping->globals = globals;
  for (i=0; i<end-start+1; i++) globals[i] = -1;
  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    globals[idx[i] - start] = i;
  }

  ierr = PetscLogObjectMemory((PetscObject)mapping,(end-start+1)*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISGlobalToLocalMappingApply"
/*@
    ISGlobalToLocalMappingApply - Provides the local numbering for a list of integers
    specified with a global numbering.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - replaces global indices with no local value with -1
           IS_GTOLM_DROP - drops the indices with no local value from the output list
.   n - number of global indices to map
-   idx - global indices to map

    Output Parameters:
+   nout - number of indices in output array (if type == IS_GTOLM_MASK then nout = n)
-   idxout - local index of each global index, one must pass in an array long enough
             to hold all the indices. You can call ISGlobalToLocalMappingApply() with
             idxout == NULL to determine the required length (returned in nout)
             and then allocate the required space and call ISGlobalToLocalMappingApply()
             a second time to set the values.

    Notes:
    Either nout or idxout may be NULL. idx and idxout may be identical.

    This is not scalable in memory usage. Each processor requires O(Nglobal) size
    array to compute these.

    Level: advanced

    Developer Note: The manual page states that idx and idxout may be identical but the calling
       sequence declares idx as const so it cannot be the same as idxout.

    Concepts: mapping^global to local

.seealso: ISLocalToGlobalMappingApply(), ISGlobalToLocalMappingApplyBlock(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
PetscErrorCode  ISGlobalToLocalMappingApply(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingType type,
                                            PetscInt n,const PetscInt idx[],PetscInt *nout,PetscInt idxout[])
{
  PetscInt       i,*globals,nf = 0,tmp,start,end,bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!mapping->globals) {
    ierr = ISGlobalToLocalMappingSetUp_Private(mapping);CHKERRQ(ierr);
  }
  globals = mapping->globals;
  start   = mapping->globalstart;
  end     = mapping->globalend;
  bs      = mapping->bs;

  if (type == IS_GTOLM_MASK) {
    if (idxout) {
      for (i=0; i<n; i++) {
        if (idx[i] < 0)                   idxout[i] = idx[i];
        else if (idx[i] < bs*start)       idxout[i] = -1;
        else if (idx[i] > bs*(end+1)-1)   idxout[i] = -1;
        else                              idxout[i] = bs*globals[idx[i]/bs - start] + (idx[i] % bs);
      }
    }
    if (nout) *nout = n;
  } else {
    if (idxout) {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) continue;
        if (idx[i] < bs*start) continue;
        if (idx[i] > bs*(end+1)-1) continue;
        tmp = bs*globals[idx[i]/bs - start] + (idx[i] % bs);
        if (tmp < 0) continue;
        idxout[nf++] = tmp;
      }
    } else {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) continue;
        if (idx[i] < bs*start) continue;
        if (idx[i] > bs*(end+1)-1) continue;
        tmp = bs*globals[idx[i]/bs - start] + (idx[i] % bs);
        if (tmp < 0) continue;
        nf++;
      }
    }
    if (nout) *nout = nf;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISGlobalToLocalMappingApplyIS"
/*@
    ISGlobalToLocalMappingApplyIS - Creates from an IS in the global numbering
    a new index set using the local numbering defined in an ISLocalToGlobalMapping
    context.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
-   is - index set in global numbering

    Output Parameters:
.   newis - index set in local numbering

    Level: advanced

    Concepts: mapping^local to global

.seealso: ISGlobalToLocalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
PetscErrorCode  ISGlobalToLocalMappingApplyIS(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingType type, IS is,IS *newis)
{
  PetscErrorCode ierr;
  PetscInt       n,nout,*idxout;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,3);
  PetscValidPointer(newis,4);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idxin);CHKERRQ(ierr);
  if (type == IS_GTOLM_MASK) {
    ierr = PetscMalloc1(n,&idxout);CHKERRQ(ierr);
  } else {
    ierr = ISGlobalToLocalMappingApply(mapping,type,n,idxin,&nout,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(nout,&idxout);CHKERRQ(ierr);
  }
  ierr = ISGlobalToLocalMappingApply(mapping,type,n,idxin,&nout,idxout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&idxin);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is),nout,idxout,PETSC_OWN_POINTER,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISGlobalToLocalMappingApplyBlock"
/*@
    ISGlobalToLocalMappingApplyBlock - Provides the local block numbering for a list of integers
    specified with a block global numbering.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - replaces global indices with no local value with -1
           IS_GTOLM_DROP - drops the indices with no local value from the output list
.   n - number of global indices to map
-   idx - global indices to map

    Output Parameters:
+   nout - number of indices in output array (if type == IS_GTOLM_MASK then nout = n)
-   idxout - local index of each global index, one must pass in an array long enough
             to hold all the indices. You can call ISGlobalToLocalMappingApplyBlock() with
             idxout == NULL to determine the required length (returned in nout)
             and then allocate the required space and call ISGlobalToLocalMappingApplyBlock()
             a second time to set the values.

    Notes:
    Either nout or idxout may be NULL. idx and idxout may be identical.

    This is not scalable in memory usage. Each processor requires O(Nglobal) size
    array to compute these.

    Level: advanced

    Developer Note: The manual page states that idx and idxout may be identical but the calling
       sequence declares idx as const so it cannot be the same as idxout.

    Concepts: mapping^global to local

.seealso: ISLocalToGlobalMappingApply(), ISGlobalToLocalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
PetscErrorCode  ISGlobalToLocalMappingApplyBlock(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingType type,
                                  PetscInt n,const PetscInt idx[],PetscInt *nout,PetscInt idxout[])
{
  PetscInt       i,*globals,nf = 0,tmp,start,end;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!mapping->globals) {
    ierr = ISGlobalToLocalMappingSetUp_Private(mapping);CHKERRQ(ierr);
  }
  globals = mapping->globals;
  start   = mapping->globalstart;
  end     = mapping->globalend;

  if (type == IS_GTOLM_MASK) {
    if (idxout) {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) idxout[i] = idx[i];
        else if (idx[i] < start) idxout[i] = -1;
        else if (idx[i] > end)   idxout[i] = -1;
        else                     idxout[i] = globals[idx[i] - start];
      }
    }
    if (nout) *nout = n;
  } else {
    if (idxout) {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) continue;
        if (idx[i] < start) continue;
        if (idx[i] > end) continue;
        tmp = globals[idx[i] - start];
        if (tmp < 0) continue;
        idxout[nf++] = tmp;
      }
    } else {
      for (i=0; i<n; i++) {
        if (idx[i] < 0) continue;
        if (idx[i] < start) continue;
        if (idx[i] > end) continue;
        tmp = globals[idx[i] - start];
        if (tmp < 0) continue;
        nf++;
      }
    }
    if (nout) *nout = nf;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingGetBlockInfo"
/*@C
    ISLocalToGlobalMappingGetBlockInfo - Gets the neighbor information for each processor and
     each index shared by more than one processor

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each subdomain (processor)
-   indices - indices of nodes (in local numbering) shared with neighbors (sorted by global numbering)

    Level: advanced

    Concepts: mapping^local to global

    Fortran Usage:
$        ISLocalToGlobalMpngGetInfoSize(ISLocalToGlobalMapping,PetscInt nproc,PetscInt numprocmax,ierr) followed by
$        ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt nproc, PetscInt procs[nproc],PetscInt numprocs[nproc],
          PetscInt indices[nproc][numprocmax],ierr)
        There is no ISLocalToGlobalMappingRestoreInfo() in Fortran. You must make sure that procs[], numprocs[] and
        indices[][] are large enough arrays, either by allocating them dynamically or defining static ones large enough.


.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingRestoreInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (mapping->info_cached) {
    *nproc = mapping->info_nproc;
    *procs = mapping->info_procs;
    *numprocs = mapping->info_numprocs;
    *indices = mapping->info_indices;
  } else {
    ierr = ISLocalToGlobalMappingGetBlockInfo_Private(mapping,nproc,procs,numprocs,indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingGetBlockInfo_Private"
static PetscErrorCode  ISLocalToGlobalMappingGetBlockInfo_Private(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag1,tag2,tag3,*len,*source,imdex;
  PetscInt       i,n = mapping->n,Ng,ng,max = 0,*lindices = mapping->indices;
  PetscInt       *nprocs,*owner,nsends,*sends,j,*starts,nmax,nrecvs,*recvs,proc;
  PetscInt       cnt,scale,*ownedsenders,*nownedsenders,rstart,nowned;
  PetscInt       node,nownedm,nt,*sends2,nsends2,*starts2,*lens2,*dest,nrecvs2,*starts3,*recvs2,k,*bprocs,*tmp;
  PetscInt       first_procs,first_numprocs,*first_indices;
  MPI_Request    *recv_waits,*send_waits;
  MPI_Status     recv_status,*send_status,*recv_statuses;
  MPI_Comm       comm;
  PetscBool      debug = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mapping,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (size == 1) {
    *nproc         = 0;
    *procs         = NULL;
    ierr           = PetscMalloc(sizeof(PetscInt),numprocs);CHKERRQ(ierr);
    (*numprocs)[0] = 0;
    ierr           = PetscMalloc(sizeof(PetscInt*),indices);CHKERRQ(ierr);
    (*indices)[0]  = NULL;
    /* save info for reuse */
    mapping->info_nproc = *nproc;
    mapping->info_procs = *procs;
    mapping->info_numprocs = *numprocs;
    mapping->info_indices = *indices;
    mapping->info_cached = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetBool(((PetscObject)mapping)->options,NULL,"-islocaltoglobalmappinggetinfo_debug",&debug,NULL);CHKERRQ(ierr);

  /*
    Notes on ISLocalToGlobalMappingGetBlockInfo

    globally owned node - the nodes that have been assigned to this processor in global
           numbering, just for this routine.

    nontrivial globally owned node - node assigned to this processor that is on a subdomain
           boundary (i.e. is has more than one local owner)

    locally owned node - node that exists on this processors subdomain

    nontrivial locally owned node - node that is not in the interior (i.e. has more than one
           local subdomain
  */
  ierr = PetscObjectGetNewTag((PetscObject)mapping,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)mapping,&tag2);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)mapping,&tag3);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    if (lindices[i] > max) max = lindices[i];
  }
  ierr   = MPIU_Allreduce(&max,&Ng,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  Ng++;
  ierr   = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  scale  = Ng/size + 1;
  ng     = scale; if (rank == size-1) ng = Ng - scale*(size-1); ng = PetscMax(1,ng);
  rstart = scale*rank;

  /* determine ownership ranges of global indices */
  ierr = PetscMalloc1(2*size,&nprocs);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);

  /* determine owners of each local node  */
  ierr = PetscMalloc1(n,&owner);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    proc             = lindices[i]/scale; /* processor that globally owns this index */
    nprocs[2*proc+1] = 1;                 /* processor globally owns at least one of ours */
    owner[i]         = proc;
    nprocs[2*proc]++;                     /* count of how many that processor globally owns of ours */
  }
  nsends = 0; for (i=0; i<size; i++) nsends += nprocs[2*i+1];
  ierr = PetscInfo1(mapping,"Number of global owners for my local data %D\n",nsends);CHKERRQ(ierr);

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);
  ierr = PetscInfo1(mapping,"Number of local owners for my global data %D\n",nrecvs);CHKERRQ(ierr);

  /* post receives for owned rows */
  ierr = PetscMalloc1((2*nrecvs+1)*(nmax+1),&recvs);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrecvs+1,&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(recvs+2*nmax*i,2*nmax,MPIU_INT,MPI_ANY_SOURCE,tag1,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* pack messages containing lists of local nodes to owners */
  ierr      = PetscMalloc1(2*n+1,&sends);CHKERRQ(ierr);
  ierr      = PetscMalloc1(size+1,&starts);CHKERRQ(ierr);
  starts[0] = 0;
  for (i=1; i<size; i++) starts[i] = starts[i-1] + 2*nprocs[2*i-2];
  for (i=0; i<n; i++) {
    sends[starts[owner[i]]++] = lindices[i];
    sends[starts[owner[i]]++] = i;
  }
  ierr = PetscFree(owner);CHKERRQ(ierr);
  starts[0] = 0;
  for (i=1; i<size; i++) starts[i] = starts[i-1] + 2*nprocs[2*i-2];

  /* send the messages */
  ierr = PetscMalloc1(nsends+1,&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsends+1,&dest);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i]) {
      ierr      = MPI_Isend(sends+starts[i],2*nprocs[2*i],MPIU_INT,i,tag1,comm,send_waits+cnt);CHKERRQ(ierr);
      dest[cnt] = i;
      cnt++;
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  /* wait on receives */
  ierr = PetscMalloc1(nrecvs+1,&source);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrecvs+1,&len);CHKERRQ(ierr);
  cnt  = nrecvs;
  ierr = PetscMalloc1(ng+1,&nownedsenders);CHKERRQ(ierr);
  ierr = PetscMemzero(nownedsenders,ng*sizeof(PetscInt));CHKERRQ(ierr);
  while (cnt) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr          = MPI_Get_count(&recv_status,MPIU_INT,&len[imdex]);CHKERRQ(ierr);
    source[imdex] = recv_status.MPI_SOURCE;
    len[imdex]    = len[imdex]/2;
    /* count how many local owners for each of my global owned indices */
    for (i=0; i<len[imdex]; i++) nownedsenders[recvs[2*imdex*nmax+2*i]-rstart]++;
    cnt--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);

  /* count how many globally owned indices are on an edge multiplied by how many processors own them. */
  nowned  = 0;
  nownedm = 0;
  for (i=0; i<ng; i++) {
    if (nownedsenders[i] > 1) {nownedm += nownedsenders[i]; nowned++;}
  }

  /* create single array to contain rank of all local owners of each globally owned index */
  ierr      = PetscMalloc1(nownedm+1,&ownedsenders);CHKERRQ(ierr);
  ierr      = PetscMalloc1(ng+1,&starts);CHKERRQ(ierr);
  starts[0] = 0;
  for (i=1; i<ng; i++) {
    if (nownedsenders[i-1] > 1) starts[i] = starts[i-1] + nownedsenders[i-1];
    else starts[i] = starts[i-1];
  }

  /* for each nontrival globally owned node list all arriving processors */
  for (i=0; i<nrecvs; i++) {
    for (j=0; j<len[i]; j++) {
      node = recvs[2*i*nmax+2*j]-rstart;
      if (nownedsenders[node] > 1) ownedsenders[starts[node]++] = source[i];
    }
  }

  if (debug) { /* -----------------------------------  */
    starts[0] = 0;
    for (i=1; i<ng; i++) {
      if (nownedsenders[i-1] > 1) starts[i] = starts[i-1] + nownedsenders[i-1];
      else starts[i] = starts[i-1];
    }
    for (i=0; i<ng; i++) {
      if (nownedsenders[i] > 1) {
        ierr = PetscSynchronizedPrintf(comm,"[%d] global node %D local owner processors: ",rank,i+rstart);CHKERRQ(ierr);
        for (j=0; j<nownedsenders[i]; j++) {
          ierr = PetscSynchronizedPrintf(comm,"%D ",ownedsenders[starts[i]+j]);CHKERRQ(ierr);
        }
        ierr = PetscSynchronizedPrintf(comm,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT);CHKERRQ(ierr);
  } /* -----------------------------------  */

  /* wait on original sends */
  if (nsends) {
    ierr = PetscMalloc1(nsends,&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(sends);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);

  /* pack messages to send back to local owners */
  starts[0] = 0;
  for (i=1; i<ng; i++) {
    if (nownedsenders[i-1] > 1) starts[i] = starts[i-1] + nownedsenders[i-1];
    else starts[i] = starts[i-1];
  }
  nsends2 = nrecvs;
  ierr    = PetscMalloc1(nsends2+1,&nprocs);CHKERRQ(ierr); /* length of each message */
  for (i=0; i<nrecvs; i++) {
    nprocs[i] = 1;
    for (j=0; j<len[i]; j++) {
      node = recvs[2*i*nmax+2*j]-rstart;
      if (nownedsenders[node] > 1) nprocs[i] += 2 + nownedsenders[node];
    }
  }
  nt = 0;
  for (i=0; i<nsends2; i++) nt += nprocs[i];

  ierr = PetscMalloc1(nt+1,&sends2);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsends2+1,&starts2);CHKERRQ(ierr);

  starts2[0] = 0;
  for (i=1; i<nsends2; i++) starts2[i] = starts2[i-1] + nprocs[i-1];
  /*
     Each message is 1 + nprocs[i] long, and consists of
       (0) the number of nodes being sent back
       (1) the local node number,
       (2) the number of processors sharing it,
       (3) the processors sharing it
  */
  for (i=0; i<nsends2; i++) {
    cnt = 1;
    sends2[starts2[i]] = 0;
    for (j=0; j<len[i]; j++) {
      node = recvs[2*i*nmax+2*j]-rstart;
      if (nownedsenders[node] > 1) {
        sends2[starts2[i]]++;
        sends2[starts2[i]+cnt++] = recvs[2*i*nmax+2*j+1];
        sends2[starts2[i]+cnt++] = nownedsenders[node];
        ierr = PetscMemcpy(&sends2[starts2[i]+cnt],&ownedsenders[starts[node]],nownedsenders[node]*sizeof(PetscInt));CHKERRQ(ierr);
        cnt += nownedsenders[node];
      }
    }
  }

  /* receive the message lengths */
  nrecvs2 = nsends;
  ierr    = PetscMalloc1(nrecvs2+1,&lens2);CHKERRQ(ierr);
  ierr    = PetscMalloc1(nrecvs2+1,&starts3);CHKERRQ(ierr);
  ierr    = PetscMalloc1(nrecvs2+1,&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs2; i++) {
    ierr = MPI_Irecv(&lens2[i],1,MPIU_INT,dest[i],tag2,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* send the message lengths */
  for (i=0; i<nsends2; i++) {
    ierr = MPI_Send(&nprocs[i],1,MPIU_INT,source[i],tag2,comm);CHKERRQ(ierr);
  }

  /* wait on receives of lens */
  if (nrecvs2) {
    ierr = PetscMalloc1(nrecvs2,&recv_statuses);CHKERRQ(ierr);
    ierr = MPI_Waitall(nrecvs2,recv_waits,recv_statuses);CHKERRQ(ierr);
    ierr = PetscFree(recv_statuses);CHKERRQ(ierr);
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);

  starts3[0] = 0;
  nt         = 0;
  for (i=0; i<nrecvs2-1; i++) {
    starts3[i+1] = starts3[i] + lens2[i];
    nt          += lens2[i];
  }
  if (nrecvs2) nt += lens2[nrecvs2-1];

  ierr = PetscMalloc1(nt+1,&recvs2);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrecvs2+1,&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs2; i++) {
    ierr = MPI_Irecv(recvs2+starts3[i],lens2[i],MPIU_INT,dest[i],tag3,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* send the messages */
  ierr = PetscMalloc1(nsends2+1,&send_waits);CHKERRQ(ierr);
  for (i=0; i<nsends2; i++) {
    ierr = MPI_Isend(sends2+starts2[i],nprocs[i],MPIU_INT,source[i],tag3,comm,send_waits+i);CHKERRQ(ierr);
  }

  /* wait on receives */
  if (nrecvs2) {
    ierr = PetscMalloc1(nrecvs2,&recv_statuses);CHKERRQ(ierr);
    ierr = MPI_Waitall(nrecvs2,recv_waits,recv_statuses);CHKERRQ(ierr);
    ierr = PetscFree(recv_statuses);CHKERRQ(ierr);
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);

  if (debug) { /* -----------------------------------  */
    cnt = 0;
    for (i=0; i<nrecvs2; i++) {
      nt = recvs2[cnt++];
      for (j=0; j<nt; j++) {
        ierr = PetscSynchronizedPrintf(comm,"[%d] local node %D number of subdomains %D: ",rank,recvs2[cnt],recvs2[cnt+1]);CHKERRQ(ierr);
        for (k=0; k<recvs2[cnt+1]; k++) {
          ierr = PetscSynchronizedPrintf(comm,"%D ",recvs2[cnt+2+k]);CHKERRQ(ierr);
        }
        cnt += 2 + recvs2[cnt+1];
        ierr = PetscSynchronizedPrintf(comm,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT);CHKERRQ(ierr);
  } /* -----------------------------------  */

  /* count number subdomains for each local node */
  ierr = PetscMalloc1(size,&nprocs);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,size*sizeof(PetscInt));CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<nrecvs2; i++) {
    nt = recvs2[cnt++];
    for (j=0; j<nt; j++) {
      for (k=0; k<recvs2[cnt+1]; k++) nprocs[recvs2[cnt+2+k]]++;
      cnt += 2 + recvs2[cnt+1];
    }
  }
  nt = 0; for (i=0; i<size; i++) nt += (nprocs[i] > 0);
  *nproc    = nt;
  ierr = PetscMalloc1(nt+1,procs);CHKERRQ(ierr);
  ierr = PetscMalloc1(nt+1,numprocs);CHKERRQ(ierr);
  ierr = PetscMalloc1(nt+1,indices);CHKERRQ(ierr);
  for (i=0;i<nt+1;i++) (*indices)[i]=NULL;
  ierr = PetscMalloc1(size,&bprocs);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i] > 0) {
      bprocs[i]        = cnt;
      (*procs)[cnt]    = i;
      (*numprocs)[cnt] = nprocs[i];
      ierr             = PetscMalloc1(nprocs[i],&(*indices)[cnt]);CHKERRQ(ierr);
      cnt++;
    }
  }

  /* make the list of subdomains for each nontrivial local node */
  ierr = PetscMemzero(*numprocs,nt*sizeof(PetscInt));CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<nrecvs2; i++) {
    nt = recvs2[cnt++];
    for (j=0; j<nt; j++) {
      for (k=0; k<recvs2[cnt+1]; k++) (*indices)[bprocs[recvs2[cnt+2+k]]][(*numprocs)[bprocs[recvs2[cnt+2+k]]]++] = recvs2[cnt];
      cnt += 2 + recvs2[cnt+1];
    }
  }
  ierr = PetscFree(bprocs);CHKERRQ(ierr);
  ierr = PetscFree(recvs2);CHKERRQ(ierr);

  /* sort the node indexing by their global numbers */
  nt = *nproc;
  for (i=0; i<nt; i++) {
    ierr = PetscMalloc1((*numprocs)[i],&tmp);CHKERRQ(ierr);
    for (j=0; j<(*numprocs)[i]; j++) tmp[j] = lindices[(*indices)[i][j]];
    ierr = PetscSortIntWithArray((*numprocs)[i],tmp,(*indices)[i]);CHKERRQ(ierr);
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  }

  if (debug) { /* -----------------------------------  */
    nt = *nproc;
    for (i=0; i<nt; i++) {
      ierr = PetscSynchronizedPrintf(comm,"[%d] subdomain %D number of indices %D: ",rank,(*procs)[i],(*numprocs)[i]);CHKERRQ(ierr);
      for (j=0; j<(*numprocs)[i]; j++) {
        ierr = PetscSynchronizedPrintf(comm,"%D ",(*indices)[i][j]);CHKERRQ(ierr);
      }
      ierr = PetscSynchronizedPrintf(comm,"\n");CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT);CHKERRQ(ierr);
  } /* -----------------------------------  */

  /* wait on sends */
  if (nsends2) {
    ierr = PetscMalloc1(nsends2,&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends2,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }

  ierr = PetscFree(starts3);CHKERRQ(ierr);
  ierr = PetscFree(dest);CHKERRQ(ierr);
  ierr = PetscFree(send_waits);CHKERRQ(ierr);

  ierr = PetscFree(nownedsenders);CHKERRQ(ierr);
  ierr = PetscFree(ownedsenders);CHKERRQ(ierr);
  ierr = PetscFree(starts);CHKERRQ(ierr);
  ierr = PetscFree(starts2);CHKERRQ(ierr);
  ierr = PetscFree(lens2);CHKERRQ(ierr);

  ierr = PetscFree(source);CHKERRQ(ierr);
  ierr = PetscFree(len);CHKERRQ(ierr);
  ierr = PetscFree(recvs);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
  ierr = PetscFree(sends2);CHKERRQ(ierr);

  /* put the information about myself as the first entry in the list */
  first_procs    = (*procs)[0];
  first_numprocs = (*numprocs)[0];
  first_indices  = (*indices)[0];
  for (i=0; i<*nproc; i++) {
    if ((*procs)[i] == rank) {
      (*procs)[0]    = (*procs)[i];
      (*numprocs)[0] = (*numprocs)[i];
      (*indices)[0]  = (*indices)[i];
      (*procs)[i]    = first_procs;
      (*numprocs)[i] = first_numprocs;
      (*indices)[i]  = first_indices;
      break;
    }
  }

  /* save info for reuse */
  mapping->info_nproc = *nproc;
  mapping->info_procs = *procs;
  mapping->info_numprocs = *numprocs;
  mapping->info_indices = *indices;
  mapping->info_cached = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingRestoreBlockInfo"
/*@C
    ISLocalToGlobalMappingRestoreBlockInfo - Frees the memory allocated by ISLocalToGlobalMappingGetBlockInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each processor
-   indices - indices of local nodes shared with neighbor (sorted by global numbering)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreBlockInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (mapping->info_free) {
    ierr = PetscFree(*numprocs);CHKERRQ(ierr);
    if (*indices) {
      PetscInt i;

      ierr = PetscFree((*indices)[0]);CHKERRQ(ierr);
      for (i=1; i<*nproc; i++) {
        ierr = PetscFree((*indices)[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(*indices);CHKERRQ(ierr);
    }
  }
  *nproc = 0;
  *procs = NULL;
  *numprocs = NULL;
  *indices = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingGetInfo"
/*@C
    ISLocalToGlobalMappingGetInfo - Gets the neighbor information for each processor and
     each index shared by more than one processor

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each subdomain (processor)
-   indices - indices of nodes (in local numbering) shared with neighbors (sorted by global numbering)

    Level: advanced

    Concepts: mapping^local to global

    Fortran Usage:
$        ISLocalToGlobalMpngGetInfoSize(ISLocalToGlobalMapping,PetscInt nproc,PetscInt numprocmax,ierr) followed by
$        ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt nproc, PetscInt procs[nproc],PetscInt numprocs[nproc],
          PetscInt indices[nproc][numprocmax],ierr)
        There is no ISLocalToGlobalMappingRestoreInfo() in Fortran. You must make sure that procs[], numprocs[] and
        indices[][] are large enough arrays, either by allocating them dynamically or defining static ones large enough.


.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingRestoreInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;
  PetscInt       **bindices = NULL,*bnumprocs = NULL,bs = mapping->bs,i,j,k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingGetBlockInfo(mapping,nproc,procs,&bnumprocs,&bindices);CHKERRQ(ierr);
  if (bs > 1) { /* we need to expand the cached info */
    ierr = PetscCalloc1(*nproc,&*indices);CHKERRQ(ierr);
    ierr = PetscCalloc1(*nproc,&*numprocs);CHKERRQ(ierr);
    for (i=0; i<*nproc; i++) {
      ierr = PetscMalloc1(bs*bnumprocs[i],&(*indices)[i]);CHKERRQ(ierr);
      for (j=0; j<bnumprocs[i]; j++) {
        for (k=0; k<bs; k++) {
          (*indices)[i][j*bs+k] = bs*bindices[i][j] + k;
        }
      }
      (*numprocs)[i] = bnumprocs[i]*bs;
    }
    mapping->info_free = PETSC_TRUE;
  } else {
    *numprocs = bnumprocs;
    *indices  = bindices;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingRestoreInfo"
/*@C
    ISLocalToGlobalMappingRestoreInfo - Frees the memory allocated by ISLocalToGlobalMappingGetInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each processor
-   indices - indices of local nodes shared with neighbor (sorted by global numbering)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingRestoreBlockInfo(mapping,nproc,procs,numprocs,indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingGetIndices"
/*@C
   ISLocalToGlobalMappingGetIndices - Get global indices for every local point that is mapped

   Not Collective

   Input Arguments:
. ltog - local to global mapping

   Output Arguments:
. array - array of indices, the length of this array may be obtained with ISLocalToGlobalMappingGetSize()

   Level: advanced

   Notes: ISLocalToGlobalMappingGetSize() returns the length the this array

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingRestoreIndices(), ISLocalToGlobalMappingGetBlockIndices(), ISLocalToGlobalMappingRestoreBlockIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  if (ltog->bs == 1) {
    *array = ltog->indices;
  } else {
    PetscInt       *jj,k,i,j,n = ltog->n, bs = ltog->bs;
    const PetscInt *ii;
    PetscErrorCode ierr;

    ierr = PetscMalloc1(bs*n,&jj);CHKERRQ(ierr);
    *array = jj;
    k    = 0;
    ii   = ltog->indices;
    for (i=0; i<n; i++)
      for (j=0; j<bs; j++)
        jj[k++] = bs*ii[i] + j;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingRestoreIndices"
/*@C
   ISLocalToGlobalMappingRestoreIndices - Restore indices obtained with ISLocalToGlobalMappingRestoreIndices()

   Not Collective

   Input Arguments:
+ ltog - local to global mapping
- array - array of indices

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingGetIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  if (ltog->bs == 1 && *array != ltog->indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Trying to return mismatched pointer");

  if (ltog->bs > 1) {
    PetscErrorCode ierr;
    ierr = PetscFree(*(void**)array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingGetBlockIndices"
/*@C
   ISLocalToGlobalMappingGetBlockIndices - Get global indices for every local block

   Not Collective

   Input Arguments:
. ltog - local to global mapping

   Output Arguments:
. array - array of indices

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingRestoreBlockIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  *array = ltog->indices;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingRestoreBlockIndices"
/*@C
   ISLocalToGlobalMappingRestoreBlockIndices - Restore indices obtained with ISLocalToGlobalMappingGetBlockIndices()

   Not Collective

   Input Arguments:
+ ltog - local to global mapping
- array - array of indices

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingGetIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreBlockIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  if (*array != ltog->indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Trying to return mismatched pointer");
  *array = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingConcatenate"
/*@C
   ISLocalToGlobalMappingConcatenate - Create a new mapping that concatenates a list of mappings

   Not Collective

   Input Arguments:
+ comm - communicator for the new mapping, must contain the communicator of every mapping to concatenate
. n - number of mappings to concatenate
- ltogs - local to global mappings

   Output Arguments:
. ltogcat - new mapping

   Note: this currently always returns a mapping with block size of 1

   Developer Note: If all the input mapping have the same block size we could easily handle that as a special case

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode ISLocalToGlobalMappingConcatenate(MPI_Comm comm,PetscInt n,const ISLocalToGlobalMapping ltogs[],ISLocalToGlobalMapping *ltogcat)
{
  PetscInt       i,cnt,m,*idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Must have a non-negative number of mappings, given %D",n);
  if (n > 0) PetscValidPointer(ltogs,3);
  for (i=0; i<n; i++) PetscValidHeaderSpecific(ltogs[i],IS_LTOGM_CLASSID,3);
  PetscValidPointer(ltogcat,4);
  for (cnt=0,i=0; i<n; i++) {
    ierr = ISLocalToGlobalMappingGetSize(ltogs[i],&m);CHKERRQ(ierr);
    cnt += m;
  }
  ierr = PetscMalloc1(cnt,&idx);CHKERRQ(ierr);
  for (cnt=0,i=0; i<n; i++) {
    const PetscInt *subidx;
    ierr = ISLocalToGlobalMappingGetSize(ltogs[i],&m);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltogs[i],&subidx);CHKERRQ(ierr);
    ierr = PetscMemcpy(&idx[cnt],subidx,m*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(ltogs[i],&subidx);CHKERRQ(ierr);
    cnt += m;
  }
  ierr = ISLocalToGlobalMappingCreate(comm,1,cnt,idx,PETSC_OWN_POINTER,ltogcat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


