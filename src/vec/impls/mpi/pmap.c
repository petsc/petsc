/*$Id: pmap.c,v 1.19 2001/01/15 21:45:04 bsmith Exp balay $*/

/*
   This file contains routines for basic map object implementation.
*/

#include "petsc.h"
#include "src/vec/vecimpl.h"   /*I  "petscvec.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "MapGetLocalSize_MPI"
int MapGetLocalSize_MPI(Map m,int *n)
{
  PetscFunctionBegin;
  *n = m->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MapGetSize_MPI"
int MapGetSize_MPI(Map m,int *N)
{
  PetscFunctionBegin;
  *N = m->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MapGetLocalRange_MPI"
int MapGetLocalRange_MPI(Map m,int *rstart,int *rend)
{
  PetscFunctionBegin;
  if (rstart) *rstart = m->rstart;
  if (rend)   *rend   = m->rend;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MapGetGlobalRange_MPI"
int MapGetGlobalRange_MPI(Map m,int *range[])
{
  PetscFunctionBegin;
  *range = m->range;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MapDestroy_MPI"
int MapDestroy_MPI(Map m)
{
  int ierr;

  PetscFunctionBegin;
  if (--m->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree(m->range);CHKERRQ(ierr);
  PetscLogObjectDestroy(m);
  PetscHeaderDestroy(m);
  PetscFunctionReturn(0);
}

static struct _MapOps DvOps = { 
            MapGetLocalSize_MPI,
            MapGetSize_MPI,
            MapGetLocalRange_MPI,
            MapGetGlobalRange_MPI,
            MapDestroy_MPI};

#undef __FUNCT__  
#define __FUNCT__ "MapCreateMPI"
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

   Concepts: maps^creating

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
  PetscLogObjectCreate(m);
  PetscLogObjectMemory(m,sizeof(struct _p_Map));
  ierr = PetscMemcpy(m->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(int),&m->range);CHKERRQ(ierr);
  
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







