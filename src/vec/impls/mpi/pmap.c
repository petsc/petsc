/*$Id: pmap.c,v 1.21 2001/07/20 21:18:16 bsmith Exp $*/

/*
   This file contains routines for basic map object implementation.
*/

#include "petsc.h"
#include "src/vec/vecimpl.h"   /*I  "petscvec.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetLocalSize_MPI"
int PetscMapGetLocalSize_MPI(PetscMap m,int *n)
{
  PetscFunctionBegin;
  *n = m->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetSize_MPI"
int PetscMapGetSize_MPI(PetscMap m,int *N)
{
  PetscFunctionBegin;
  *N = m->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetLocalRange_MPI"
int PetscMapGetLocalRange_MPI(PetscMap m,int *rstart,int *rend)
{
  PetscFunctionBegin;
  if (rstart) *rstart = m->rstart;
  if (rend)   *rend   = m->rend;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetGlobalRange_MPI"
int PetscMapGetGlobalRange_MPI(PetscMap m,int *range[])
{
  PetscFunctionBegin;
  *range = m->range;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapDestroy_MPI"
int PetscMapDestroy_MPI(PetscMap m)
{
  int ierr;

  PetscFunctionBegin;
  if (--m->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree(m->range);CHKERRQ(ierr);
  PetscLogObjectDestroy(m);
  PetscHeaderDestroy(m);
  PetscFunctionReturn(0);
}

static struct _PetscMapOps DvOps = { 
            PetscMapGetLocalSize_MPI,
            PetscMapGetSize_MPI,
            PetscMapGetLocalRange_MPI,
            PetscMapGetGlobalRange_MPI,
            PetscMapDestroy_MPI};

#undef __FUNCT__  
#define __FUNCT__ "PetscMapCreateMPI"
/*@C
   PetscMapCreateMPI - Creates a map object.

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

.seealso: PetscMapDestroy(), PetscMapGetLocalSize(), PetscMapGetSize(), PetscMapGetGlobalRange(),
          PetscMapGetLocalRange()

@*/ 
int PetscMapCreateMPI(MPI_Comm comm,int n,int N,PetscMap *mm)
{
  int      ierr,i,rank,size;
  PetscMap m;

  PetscFunctionBegin;
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 

  PetscHeaderCreate(m,_p_PetscMap,struct _PetscMapOps,MAP_COOKIE,0,"PetscMap",comm,PetscMapDestroy,0);
  PetscLogObjectCreate(m);
  PetscLogObjectMemory(m,sizeof(struct _p_PetscMap));
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







