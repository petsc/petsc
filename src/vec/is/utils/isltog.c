
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: isltog.c,v 1.14 1997/09/11 20:38:10 bsmith Exp bsmith $";
#endif

#include "sys.h"   /*I "sys.h" I*/
#include "src/is/isimpl.h"    /*I "is.h"  I*/

#undef __FUNC__  
#define __FUNC__ "ISLocalToGlobalMappingCreate"
/*@C
    ISLocalToGlobalMappingCreate - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Input Parameters:
.   comm - MPI communicator of size 1.
.   n - the number of local elements
.   indices - the global index for each local element

    Output Parameters:
.   mapping - new mapping data structure

.keywords: IS, local-to-global mapping, create

.seealso: ISLocalToGlobalMappingDestroy()
@*/
int ISLocalToGlobalMappingCreate(MPI_Comm cm,int n, int *indices,ISLocalToGlobalMapping *mapping)
{
  PetscValidIntPointer(indices);
  PetscValidPointer(mapping);

  PetscHeaderCreate(*mapping,_p_ISLocalToGlobalMapping,IS_LTOGM_COOKIE,0,cm,ISLocalToGlobalMappingDestroy,0);
  PLogObjectCreate(*mapping);
  PLogObjectMemory(*mapping,sizeof(struct _p_ISLocalToGlobalMapping)+n*sizeof(int));

  (*mapping)->n       = n;
  (*mapping)->indices = (int *) PetscMalloc((n+1)*sizeof(int));CHKPTRQ((*mapping)->indices);
  PetscMemcpy((*mapping)->indices,indices,n*sizeof(int));

  /*
      Do not create the global to local mapping. This is only created if 
     ISGlobalToLocalMapping() is called 
  */
  (*mapping)->globals = 0;
  return 0;
}
  
#undef __FUNC__  
#define __FUNC__ "ISLocalToGlobalMappingDestroy"
/*@
   ISLocalToGlobalMappingDestroy - Destroys a mapping between a local (0 to n)
   ordering and a global parallel ordering.

   Input Parameters:
.  mapping - mapping data structure

.keywords: IS, local-to-global mapping, destroy

.seealso: ISLocalToGlobalMappingCreate()
@*/
int ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping mapping)
{
  PetscValidPointer(mapping);
  if (--mapping->refct > 0) return 0;

  PetscFree(mapping->indices);
  if (mapping->globals) PetscFree(mapping->globals);
  PLogObjectDestroy(mapping);
  PetscHeaderDestroy(mapping);
  return 0;
}
  
#undef __FUNC__  
#define __FUNC__ "ISLocalToGlobalMappingApplyIS"
/*@
    ISLocalToGlobalMappingApplyIS - Creates from an IS in the local numbering
    a new index set using the global numbering defined in an ISLocalToGlobalMapping
    context.

    Input Parameters:
.   mapping - mapping between local and global numbering
.   is - index set in local numbering

    Output Parameters:
.   newis - index set in global numbering

.keywords: IS, local-to-global mapping, apply

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy(), ISGlobalToLocalMappingApply()
@*/
int ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping, IS is, IS *newis)
{
  int ierr,n,i,*idxin,*idxmap,*idxout;
  PetscValidPointer(mapping);
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidPointer(newis);

  ierr   = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr   = ISGetIndices(is,&idxin); CHKERRQ(ierr);
  idxmap = mapping->indices;
  
  idxout = (int *) PetscMalloc((n+1)*sizeof(int));CHKPTRQ(idxout);
  for ( i=0; i<n; i++ ) {
    idxout[i] = idxmap[idxin[i]];
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idxout,newis); CHKERRQ(ierr);
  PetscFree(idxout);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISLocalToGlobalMappingApply"
/*@C
   ISLocalToGlobalMappingApply - Takes a list of integers in a local numbering
   and converts them to the global numbering.

   Input Parameters:
.  mapping - the local to global mapping context
.  N - number of integers
.  in - input indices in local numbering

   Output Parameter:
.  out - indices in global numbering

   Notes: The in and out array may be identical

.seealso: ISLocalToGlobalMappingCreate(),ISLocalToGlobalMappingDestroy(), 
          ISLocalToGlobalMappingApplyIS(),AOCreateBasic(),AOApplicationToPetsc(),
          AOPetscToApplication(), ISGlobalToLocalMappingApply()

.keywords: local-to-global, mapping, apply

@*/
int ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,int N,int *in,int *out)
{
  int i,*idx = mapping->indices,Nmax = mapping->n;
  for ( i=0; i<N; i++ ) {
    if (in[i] < 0) {out[i] = in[i]; continue;}
    if (in[i] >= Nmax) SETERRQ(1,1,"Local index too large");
    out[i] = idx[in[i]];
  }
  return 0;
}

/* -----------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "ISGlobalToLocalMappingSetUp_Private"
/*
    Creates the global fields in the ISLocalToGlobalMapping structure
*/
static int ISGlobalToLocalMappingSetUp_Private(ISLocalToGlobalMapping mapping)
{
  int i,*idx = mapping->indices,n = mapping->n,end,start,*globals;

  end   = 0;
  start = 100000000;

  for ( i=0; i<n; i++ ) {
    if (idx[i] < 0) continue;
    if (idx[i] < start) start = idx[i];
    if (idx[i] > end)   end   = idx[i];
  }
  if (start > end) {start = 0; end = -1;}
  mapping->globalstart = start;
  mapping->globalend   = end;

  globals = mapping->globals = (int *) PetscMalloc((end-start+2)*sizeof(int));CHKPTRQ(mapping->globals);
  for ( i=0; i<end-start+1; i++ ) {
    globals[i] = -1;
  }
  for ( i=0; i<n; i++ ) {
    if (idx[i] < 0) continue;
    globals[idx[i] - start] = i;
  }

  PLogObjectMemory(mapping,(end-start+1)*sizeof(int));
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISGlobalToLocalMappingApply"
/*@
    ISGlobalToLocalMappingApply - Takes a list of integers in global numbering
      and returns the local numbering.

    Input Parameters:
.   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - replaces global indices with no local value with -1
           IS_GTOLM_DROP - drops the indices with no local value from the output list
.   n - number of global indices to map
.   idx - global indices to map

    Output Parameters:
.   nout - number of indices in output array (if type == IS_GTOLM_MASK then nout = n)
.   idxout - local index of each global index, one must pass in an array long enough 
             to hold all the indices. You can call ISGlobalToLocalMappingApply() with 
             idxout == PETSC_NULL to determine the required length (returned in nout)
             and then allocate the required space and call ISGlobalToLocalMappingApply()
             a second time to set the values.

    Notes: Either nout or idxout may be PETSC_NULL. idx and idxout may be identical.

.keywords: IS, global-to-local mapping, apply

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
int ISGlobalToLocalMappingApply(ISLocalToGlobalMapping mapping, ISGlobalToLocalMappingType type,
                                  int n, int *idx,int *nout,int *idxout)
{
  int i,ierr, *globals,nf = 0,tmp,start,end;

  if (!mapping->globals) {
    ierr = ISGlobalToLocalMappingSetUp_Private(mapping); CHKERRQ(ierr);
  }
  globals = mapping->globals;
  start   = mapping->globalstart;
  end     = mapping->globalend;

  if (type == IS_GTOLM_MASK) {
    if (idxout) {
      for ( i=0; i<n; i++ ) {
        if (idx[i] < 0) idxout[i] = idx[i]; 
        else if (idx[i] < start) idxout[i] = -1;
        else if (idx[i] > end)   idxout[i] = -1;
        else                     idxout[i] = globals[idx[i] - start];
      }
    }
    if (nout) *nout = n;
  } else {
    if (idxout) {
      for ( i=0; i<n; i++ ) {
        if (idx[i] < 0) continue;
        if (idx[i] < start) continue;
        if (idx[i] > end) continue;
        tmp = globals[idx[i] - start];
        if (tmp < 0) continue;
        idxout[nf++] = tmp;
      }
    } else {
      for ( i=0; i<n; i++ ) {
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

  return 0;
}

