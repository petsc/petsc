/*$Id: pmap.c,v 1.14 2000/01/11 21:00:13 bsmith Exp bsmith $*/

/*
   This file contains routines for basic map object implementation.
*/

#include "petsc.h"
#include "src/vec/vecimpl.h"   /*I  "vec.h"   I*/

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MapGetLocalSize_MPI"
int MapGetLocalSize_MPI(Map m,int *n)
{
  PetscFunctionBegin;
  *n = m->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MapGetSize_MPI"
int MapGetSize_MPI(Map m,int *N)
{
  PetscFunctionBegin;
  *N = m->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MapGetLocalRange_MPI"
int MapGetLocalRange_MPI(Map m,int *rstart,int *rend)
{
  PetscFunctionBegin;
  if (rstart) *rstart = m->rstart;
  if (rend)   *rend   = m->rend;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MapGetGlobalRange_MPI"
int MapGetGlobalRange_MPI(Map m,int *range[])
{
  PetscFunctionBegin;
  *range = m->range;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MapDestroy_MPI"
int MapDestroy_MPI(Map m)
{
  int ierr;

  PetscFunctionBegin;
  if (--m->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree(m->range);CHKERRQ(ierr);
  PLogObjectDestroy(m);
  PetscHeaderDestroy(m);
  PetscFunctionReturn(0);
}

static struct _MapOps DvOps = { 
            MapGetLocalSize_MPI,
            MapGetSize_MPI,
            MapGetLocalRange_MPI,
            MapGetGlobalRange_MPI,
            MapDestroy_MPI};

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MapCreateMPI"
/*@C
   MapCreateMPI - Creates a map object.

   Collective on MPI_Comm
 
   Input Parameters:
+  comm - the MPI communicator to use 
.  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
-  N - global vector length (or PETSC_DECIDE to have calculated if n is given)

   Output Parameter:
.  mm - the map object

   Suggested by:
   Robert Clay and Alan Williams, developers of ISIS++, Sandia National Laboratories.

   Level: developer

.keywords: Map, create, MPI

.seealso: MapDestroy(), MapGetLocalSize(), MapGetSize(), MapGetGlobalRange(),
          MapGetLocalRange()

@*/ 
int MapCreateMPI(MPI_Comm comm,int n,int N,Map *mm)
{
  int ierr,i,rank,size;
  Map m;

  PetscFunctionBegin;
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 

  PetscHeaderCreate(m,_p_Map,struct _MapOps,MAP_COOKIE,0,"Map",comm,MapDestroy,0);
  PLogObjectCreate(m);
  PLogObjectMemory(m,sizeof(struct _p_Map));
  ierr     = PetscMemcpy(m->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  m->range = (int*)PetscMalloc((size+1)*sizeof(int));CHKPTRQ(m->range);
  
  ierr = MPI_Allgather(&n,1,MPI_INT,m->range+1,1,MPI_INT,comm);CHKERRQ(ierr);
  m->range[0] = 0;
  for (i=2; i<=size; i++) {
    m->range[i] += m->range[i-1];
  }
  m->rstart = m->range[rank];
  m->rend   = m->range[rank+1];
  m->n      = n;
  m->N      = N;
  *mm = m;

  PetscFunctionReturn(0);
}







