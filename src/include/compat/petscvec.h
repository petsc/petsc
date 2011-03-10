#ifndef _COMPAT_PETSC_VEC_H
#define _COMPAT_PETSC_VEC_H

#include "private/vecimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define VECSTANDARD "standard"
#define VECSEQCUSP  "seqcusp"
#define VECMPICUSP  "mpicusp"
#define VECCUSP     "cusp"
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "VecDestroyVecs"
static PetscErrorCode VecDestroyVecs_Compat(PetscInt m,Vec *vv[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(vv,1);
  ierr = VecDestroyVecs(vv[0],m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define VecDestroyVecs VecDestroyVecs_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "VecLoad"
static PetscErrorCode VecLoad_Compat(Vec vec,PetscViewer viewer)
{
  const VecType  type;
  PetscInt       n,N;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  ierr = VecGetType(vec,&type);CHKERRQ(ierr);
  if (type) {
    ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
    ierr = VecGetSize(vec,&N);CHKERRQ(ierr);
    if (n>=0 && N>=0) {
      ierr = VecLoadIntoVector(viewer,vec);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  {
    Vec           loadvec;
    const VecType loadtype = type;
    if (!loadtype) {
      MPI_Comm    comm;
      PetscMPIInt size;
      ierr = PetscObjectGetComm((PetscObject)vec,&comm);CHKERRQ(ierr);
      ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
      loadtype = (size > 1) ? VECMPI : VECSEQ;
    }
    ierr = VecLoad(viewer,loadtype,&loadvec);
    ierr = VecGetType(loadvec,&loadtype);CHKERRQ(ierr);
    ierr = VecGetLocalSize(loadvec,&n);CHKERRQ(ierr);
    ierr = VecGetSize(loadvec,&N);CHKERRQ(ierr);
    ierr = VecSetSizes(vec,n,N);CHKERRQ(ierr);
    ierr = VecGetType(vec,&type);CHKERRQ(ierr);
    if (!type) {
      ierr = VecSetType(vec,loadtype);CHKERRQ(ierr);
    }
    ierr = VecCopy(loadvec,vec);CHKERRQ(ierr);
    ierr = VecDestroy(loadvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#define VecLoad VecLoad_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define VecSqrtAbs VecSqrt
#endif

#endif /* _COMPAT_PETSC_VEC_H */
