#define PETSCVEC_DLL
/*
   This file contains routines for basic map object implementation.
*/

#include "private/vecimpl.h"   /*I  "petscvec.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscMapInitialize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapInitialize(MPI_Comm comm,PetscInt m,PetscInt N,PetscMap *map)
{
  PetscMPIInt    rank,size;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr); 
  map->n = m;
  map->N = N;
  ierr = PetscSplitOwnership(comm,&map->n,&map->N);CHKERRQ(ierr);
  if (!map->range) {
    ierr = PetscMalloc((size+1)*sizeof(PetscInt), &map->range);CHKERRQ(ierr);
  }
  ierr = MPI_Allgather(&map->n, 1, MPIU_INT, map->range+1, 1, MPIU_INT, comm);CHKERRQ(ierr);

  map->range[0] = 0;
  for(p = 2; p <= size; p++) {
    map->range[p] += map->range[p-1];
  }

  map->rstart = map->range[rank];
  map->rend   = map->range[rank+1];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapCopy"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapCopy(MPI_Comm comm,PetscMap *in,PetscMap *out)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscInt       *range = out->range;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMemcpy(out,in,sizeof(PetscMap));CHKERRQ(ierr);
  if (!range) {
    ierr = PetscMalloc((size+1)*sizeof(PetscInt),&out->range);CHKERRQ(ierr);
  } else {
    out->range = range;
  }
  ierr = PetscMemcpy(out->range,in->range,size*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

