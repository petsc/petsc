/*$Id: isltog.c,v 1.40 2000/05/05 22:14:51 balay Exp bsmith $*/

#include "petscsys.h"   /*I "petscsys.h" I*/
#include "src/vec/is/isimpl.h"    /*I "petscis.h"  I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="ISLocalToGlobalMappingGetSize"></a>*/"ISLocalToGlobalMappingGetSize"
/*@C
    ISLocalToGlobalMappingGetSize - Gets the local size of a local to global mapping.

    Not Collective

    Input Parameter:
.   ltog - local to global mapping

    Output Parameter:
.   n - the number of entries in the local mapping

    Level: advanced

.keywords: IS, local-to-global mapping, create

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
int ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping mapping,int *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);
  *n = mapping->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ISLocalToGlobalMappingView"
/*@C
    ISLocalToGlobalMappingView - View a local to global mapping

    Not Collective

    Input Parameters:
+   ltog - local to global mapping
-   viewer - viewer

    Level: advanced

.keywords: IS, local-to-global mapping, create

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
int ISLocalToGlobalMappingView(ISLocalToGlobalMapping mapping,Viewer viewer)
{
  int        i,ierr,rank;
  PetscTruth isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);
  if (!viewer) viewer = VIEWER_STDOUT_(mapping->comm);
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  PetscCheckSameComm(mapping,viewer);

  ierr = MPI_Comm_rank(mapping->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    for (i=0; i<mapping->n; i++) {
      ierr = ViewerASCIISynchronizedPrintf(viewer,"[%d] %d %d\n",rank,i,mapping->indices[i]);CHKERRQ(ierr);
    }
    ierr = ViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for ISLocalToGlobalMapping",((PetscObject)viewer)->type_name);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ISLocalToGlobalMappingCreateIS"
/*@C
    ISLocalToGlobalMappingCreateIS - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not collective

    Input Parameter:
.   is - index set containing the global numbers for each local

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

.keywords: IS, local-to-global mapping, create

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
int ISLocalToGlobalMappingCreateIS(IS is,ISLocalToGlobalMapping *mapping)
{
  int      n,*indices,ierr;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm,n,indices,mapping);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ISLocalToGlobalMappingCreate"
/*@C
    ISLocalToGlobalMappingCreate - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not Collective 

    Input Parameters:
+   comm - MPI communicator of size 1.
.   n - the number of local elements
-   indices - the global index for each local element

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

.keywords: IS, local-to-global mapping, create

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS()
@*/
int ISLocalToGlobalMappingCreate(MPI_Comm cm,int n,const int indices[],ISLocalToGlobalMapping *mapping)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(indices);
  PetscValidPointer(mapping);

  PetscHeaderCreate(*mapping,_p_ISLocalToGlobalMapping,int,IS_LTOGM_COOKIE,0,"ISLocalToGlobalMapping",
                    cm,ISLocalToGlobalMappingDestroy,ISLocalToGlobalMappingView);
  PLogObjectCreate(*mapping);
  PLogObjectMemory(*mapping,sizeof(struct _p_ISLocalToGlobalMapping)+n*sizeof(int));

  (*mapping)->n       = n;
  (*mapping)->indices = (int*)PetscMalloc((n+1)*sizeof(int));CHKPTRQ((*mapping)->indices);
  ierr = PetscMemcpy((*mapping)->indices,indices,n*sizeof(int));CHKERRQ(ierr);

  /*
      Do not create the global to local mapping. This is only created if 
     ISGlobalToLocalMapping() is called 
  */
  (*mapping)->globals = 0;
  PetscFunctionReturn(0);
}
  
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ISLocalToGlobalMappingDestroy"
/*@
   ISLocalToGlobalMappingDestroy - Destroys a mapping between a local (0 to n)
   ordering and a global parallel ordering.

   Note Collective

   Input Parameters:
.  mapping - mapping data structure

   Level: advanced

.keywords: IS, local-to-global mapping, destroy

.seealso: ISLocalToGlobalMappingCreate()
@*/
int ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping mapping)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidPointer(mapping);
  if (--mapping->refct > 0) PetscFunctionReturn(0);
  if (mapping->refct < 0) {
    SETERRQ(1,1,"Mapping already destroyed");
  }

  ierr = PetscFree(mapping->indices);CHKERRQ(ierr);
  if (mapping->globals) {ierr = PetscFree(mapping->globals);CHKERRQ(ierr);}
  PLogObjectDestroy(mapping);
  PetscHeaderDestroy(mapping);
  PetscFunctionReturn(0);
}
  
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ISLocalToGlobalMappingApplyIS"
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

.keywords: IS, local-to-global mapping, apply

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy(), ISGlobalToLocalMappingApply()
@*/
int ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping,IS is,IS *newis)
{
  int ierr,n,i,*idxin,*idxmap,*idxout,Nmax = mapping->n;

  PetscFunctionBegin;
  PetscValidPointer(mapping);
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidPointer(newis);

  ierr   = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr   = ISGetIndices(is,&idxin);CHKERRQ(ierr);
  idxmap = mapping->indices;
  
  idxout = (int*)PetscMalloc((n+1)*sizeof(int));CHKPTRQ(idxout);
  for (i=0; i<n; i++) {
    if (idxin[i] >= Nmax) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,1,"Local index %d too large %d (max) at %d",idxin[i],Nmax,i);
    idxout[i] = idxmap[idxin[i]];
  }
  ierr = ISRestoreIndices(is,&idxin);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idxout,newis);CHKERRQ(ierr);
  ierr = PetscFree(idxout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@MC
   ISLocalToGlobalMappingApply - Takes a list of integers in a local numbering
   and converts them to the global numbering.

   Not collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local numbering

   Output Parameter:
.  out - indices in global numbering

   Synopsis:
   ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,int N,int in[],int out[])

   Notes: 
   The in and out array parameters may be identical.

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(),ISLocalToGlobalMappingDestroy(), 
          ISLocalToGlobalMappingApplyIS(),AOCreateBasic(),AOApplicationToPetsc(),
          AOPetscToApplication(), ISGlobalToLocalMappingApply()

.keywords: local-to-global, mapping, apply

@*/

/* -----------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ISGlobalToLocalMappingSetUp_Private"
/*
    Creates the global fields in the ISLocalToGlobalMapping structure
*/
static int ISGlobalToLocalMappingSetUp_Private(ISLocalToGlobalMapping mapping)
{
  int i,*idx = mapping->indices,n = mapping->n,end,start,*globals;

  PetscFunctionBegin;
  end   = 0;
  start = 100000000;

  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    if (idx[i] < start) start = idx[i];
    if (idx[i] > end)   end   = idx[i];
  }
  if (start > end) {start = 0; end = -1;}
  mapping->globalstart = start;
  mapping->globalend   = end;

  globals = mapping->globals = (int*)PetscMalloc((end-start+2)*sizeof(int));CHKPTRQ(mapping->globals);
  for (i=0; i<end-start+1; i++) {
    globals[i] = -1;
  }
  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    globals[idx[i] - start] = i;
  }

  PLogObjectMemory(mapping,(end-start+1)*sizeof(int));
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ISGlobalToLocalMappingApply"
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
             idxout == PETSC_NULL to determine the required length (returned in nout)
             and then allocate the required space and call ISGlobalToLocalMappingApply()
             a second time to set the values.

    Notes:
    Either nout or idxout may be PETSC_NULL. idx and idxout may be identical.

    This is not scalable in memory usage. Each processor requires O(Nglobal) size 
    array to compute these.

    Level: advanced

.keywords: IS, global-to-local mapping, apply

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
int ISGlobalToLocalMappingApply(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingType type,
                                  int n,const int idx[],int *nout,int idxout[])
{
  int i,ierr,*globals,nf = 0,tmp,start,end;

  PetscFunctionBegin;
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

