#define PETSCVEC_DLL
/*
   This file contains routines for basic map object implementation.
*/

#include "private/vecimpl.h"   /*I  "petscvec.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscMapDestroy_MPI"
PetscErrorCode PetscMapDestroy_MPI(PetscMap m)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static struct _PetscMapOps DvOps = {
  PETSC_NULL,
  PetscMapDestroy_MPI,
};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscMapCreate_MPI"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapCreate_MPI(PetscMap m)
{
  PetscMPIInt    rank,size;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(m->ops, &DvOps, sizeof(DvOps));CHKERRQ(ierr);

  ierr = MPI_Comm_size(m->comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(m->comm, &rank);CHKERRQ(ierr); 
  ierr = PetscSplitOwnership(m->comm,&m->n,&m->N);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(PetscInt), &m->range);CHKERRQ(ierr);
  ierr = MPI_Allgather(&m->n, 1, MPIU_INT, m->range+1, 1, MPIU_INT, m->comm);CHKERRQ(ierr);

  m->range[0] = 0;
  for(p = 2; p <= size; p++) {
    m->range[p] += m->range[p-1];
  }

  m->rstart = m->range[rank];
  m->rend   = m->range[rank+1];
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscMapCreateMPI"
/*@
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
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapCreateMPI(MPI_Comm comm,PetscInt n,PetscInt N,PetscMap *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMapCreate(comm, m);CHKERRQ(ierr);
  ierr = PetscMapSetLocalSize(*m, n);CHKERRQ(ierr);
  ierr = PetscMapSetSize(*m, N);CHKERRQ(ierr);
  ierr = PetscMapSetType(*m, MAP_MPI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
