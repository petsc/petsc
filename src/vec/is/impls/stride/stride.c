#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stride.c,v 1.66 1998/03/12 23:14:57 bsmith Exp bsmith $";
#endif
/*
       Index sets of evenly space integers, defined by a 
    start, stride and length.
*/
#include "src/is/isimpl.h"             /*I   "is.h"   I*/
#include "pinclude/pviewer.h"

typedef struct {
  int n,first,step;
} IS_Stride;

#undef __FUNC__  
#define __FUNC__ "ISDuplicate_Stride" 
int ISDuplicate_Stride(IS is, IS *newIS)
{
  int       ierr;
  IS_Stride *sub = (IS_Stride *) is->data;

  PetscFunctionBegin;
  ierr = ISCreateStride(is->comm, sub->n, sub->first, sub->step, newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISInvertPermutation_Stride" 
int ISInvertPermutation_Stride(IS is, IS *perm)
{
  IS_Stride *isstride = (IS_Stride *) is->data;
  int       ierr;

  PetscFunctionBegin;
  if (is->isidentity) {
    ierr = ISCreateStride(PETSC_COMM_SELF,isstride->n,0,1,perm); CHKERRQ(ierr);
    ierr = ISSetPermutation(*perm); CHKERRQ(ierr);
  } else {
    int *ii,*indices,i,n = isstride->n;
    ierr = ISGetIndices(is,&indices); CHKERRQ(ierr);
    ii = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(ii);
    for ( i=0; i<n; i++ ) {
      ii[indices[i]] = i;
    }
    ierr = ISRestoreIndices(is,&indices); CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,perm); CHKERRQ(ierr);
    PetscFree(ii);
    ierr = ISSetPermutation(*perm); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
    
#undef __FUNC__  
#define __FUNC__ "ISStrideGetInfo" 
/*@
   ISStrideGetInfo - Returns the first index in a stride index set and 
   the stride width.

   Input Parameter:
.  is - the index set

   Output Parameters:
.  first - the first index
.  step - the stride width

   Notes:
   Returns info on stride index set. This is a pseudo-public function that
   should not be needed by most users.

.keywords: IS, index set, stride, get, information

.seealso: ISCreateStride(), ISGetSize()
@*/
int ISStrideGetInfo(IS is,int *first,int *step)
{
  IS_Stride *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(first);
  PetscValidIntPointer(step);

  sub = (IS_Stride *) is->data;
  if (is->type != IS_STRIDE) PetscFunctionReturn(0);
  *first = sub->first; *step = sub->step;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISStride" 
/*@C
   ISStride - Determines if an IS is based on a stride.

   Input Parameter:
.  is - the index set

   Output Parameters:
.  flag - either PETSC_TRUE or PETSC_FALSE

.keywords: IS, index set, stride, get, information

.seealso: ISCreateStride(), ISGetSize()
@*/
int ISStride(IS is,PetscTruth *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(flag);

  if (is->type != IS_STRIDE) *flag = PETSC_FALSE;
  else                           *flag = PETSC_TRUE;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISDestroy_Stride" 
int ISDestroy_Stride(IS is)
{
  PetscFunctionBegin;
  PetscFree(is->data); 
  PLogObjectDestroy(is);
  PetscHeaderDestroy(is); PetscFunctionReturn(0);
}

/*
     Returns a legitimate index memory even if 
   the stride index set is empty.
*/
#undef __FUNC__  
#define __FUNC__ "ISGetIndices_Stride" 
int ISGetIndices_Stride(IS in,int **idx)
{
  IS_Stride *sub = (IS_Stride *) in->data;
  int       i;

  PetscFunctionBegin;
  *idx = (int *) PetscMalloc((sub->n+1)*sizeof(int)); CHKPTRQ(idx);
  (*idx)[0] = sub->first;
  for ( i=1; i<sub->n; i++ ) (*idx)[i] = (*idx)[i-1] + sub->step;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISRestoreIndices_Stride" 
int ISRestoreIndices_Stride(IS in,int **idx)
{
  PetscFunctionBegin;
  PetscFree(*idx);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISGetSize_Stride" 
int ISGetSize_Stride(IS is,int *size)
{
  IS_Stride *sub = (IS_Stride *)is->data;

  PetscFunctionBegin;
  *size = sub->n; PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISView_Stride" 
int ISView_Stride(IS is, Viewer viewer)
{
  IS_Stride   *sub = (IS_Stride *)is->data;
  int         i,n = sub->n,ierr,rank;
  FILE        *fd;
  ViewerType  vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (is->isperm) {
      fprintf(fd,"Index set is permutation\n");
    }
    fprintf(fd,"Number of indices in (stride) set %d\n",n);
    for ( i=0; i<n; i++ ) {
      fprintf(fd,"%d %d\n",i,sub->first + i*sub->step);
    }
    fflush(fd);
  } else if (vtype  == ASCII_FILES_VIEWER) { 
    MPI_Comm_rank(is->comm,&rank);
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (is->isperm) {
      fprintf(fd,"[%d] Index set is permutation\n",rank);
    }
    fprintf(fd,"[%d] Number of indices in (stride) set %d\n",rank,n);
    for ( i=0; i<n; i++ ) {
      fprintf(fd,"[%d] %d %d\n",rank,i,sub->first + i*sub->step);
    }
    fflush(fd);
  }
  PetscFunctionReturn(0);
}
  
#undef __FUNC__  
#define __FUNC__ "ISSort_Stride" 
int ISSort_Stride(IS is)
{
  IS_Stride *sub = (IS_Stride *) is->data;

  PetscFunctionBegin;
  if (sub->step >= 0 ) PetscFunctionReturn(0);
  sub->first += (sub->n - 1 )*sub->step;
  sub->step *= -1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSorted_Stride" 
int ISSorted_Stride(IS is, PetscTruth* flg)
{
  IS_Stride *sub = (IS_Stride *) is->data;

  PetscFunctionBegin;
  if (sub->step >= 0) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static struct _ISOps myops = { ISGetSize_Stride,
                               ISGetSize_Stride,
                               ISGetIndices_Stride,
                               ISRestoreIndices_Stride,
                               ISInvertPermutation_Stride,
                               ISSort_Stride, 
                               ISSorted_Stride,
                               ISDuplicate_Stride };

#undef __FUNC__  
#define __FUNC__ "ISCreateStride" 
/*@C
   ISCreateStride - Creates a data structure for an index set 
   containing a list of evenly spaced integers.

   Input Parameters:
.  comm - the MPI communicator
.  n - the length of the index set
.  first - the first element of the index set
.  step - the change to the next index

   Output Parameter:
.  is - the new index set

.keywords: IS, index set, create, stride

.seealso: ISCreateGeneral(), ISCreateBlock()
@*/
int ISCreateStride(MPI_Comm comm,int n,int first,int step,IS *is)
{
  int       min, max,flg,ierr;
  IS        Nindex;
  IS_Stride *sub;

  PetscFunctionBegin;
  *is = 0;
  if (n < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Number of indices < 0");

  PetscHeaderCreate(Nindex, _p_IS,struct _ISOps,IS_COOKIE,IS_STRIDE,comm,ISDestroy,ISView); 
  PLogObjectCreate(Nindex);
  PLogObjectMemory(Nindex,sizeof(IS_Stride) + sizeof(struct _p_IS));
  sub            = PetscNew(IS_Stride); CHKPTRQ(sub);
  sub->n         = n;
  sub->first     = first;
  sub->step      = step;
  if (step > 0) {min = first; max = first + step*(n-1);}
  else          {max = first; min = first + step*(n-1);}

  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void *) sub;
  PetscMemcpy(Nindex->ops,&myops,sizeof(myops));
  Nindex->ops->destroy = ISDestroy_Stride;
  Nindex->ops->view    = ISView_Stride;
  Nindex->isperm  = 0;
  ierr = OptionsHasName(PETSC_NULL,"-is_view",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = ISView(Nindex,VIEWER_STDOUT_(Nindex->comm)); CHKERRQ(ierr);
  }
  *is = Nindex; 
  PetscFunctionReturn(0);
}




