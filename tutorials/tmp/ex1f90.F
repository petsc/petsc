!
!
!    Reads in data prepared by ex1.c in parallel, does a suitable partitioning of the 
!    data and arranges it in a format suitable for Pflotran.
!

      program main
      implicit none


! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  The following include statements are required for Fortran programs
!  that use PETSc vectors:
!     petsc.h  - base PETSc routines
!     petscvec.h - defines (INSERT_VALUES)
!     petscmat.h    - matrices
!     petscmat.h90  - to allow access to Fortran 90 features of matrices

#include "finclude/petsc.h"
#include "finclude/petscis.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscviewer.h"

      Mat Adj, Adjnew
      PetscErrorCode ierr
      PetscViewer petscviewer
      MatPartitioning mp
      IS partitioning,newnumbering
      IS oldnumbering,all_oldnumbering
      IS allnewnumbering
      PetscInt nlocals(100) ! need to allocate correct size
      PetscMPIInt size,rank

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'mesh.petsc',                &
     &                           FILE_MODE_READ,petscviewer,ierr)

      call MatLoad(petscviewer,MATMPIAIJ,Adj,ierr)

      call MatPartitioningCreate(PETSC_COMM_WORLD,mp,ierr)
      call MatPartitioningSetAdjacency(mp,Adj,ierr)
      call MatPartitioningSetFromOptions(mp,ierr)
      call MatPartitioningApply(mp,partitioning,ierr)
      call MatPartitioningDestroy(mp,ierr)
!
!    newnumbering is the new cell number for each old cell number
!
      call ISPartitioningToNumbering(partitioning,newnumbering,ierr)
      call ISDestroy(partitioning,ierr)

      call ISAllGather(newnumbering,allnewnumbering,ierr)
      call MatPermute(Adj,newnumbering,allnewnumbering,Adjnew,ierr)

      call ISView(newnumbering,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatDestroy(Adjnew,ierr)
      call ISDestroy(newnumbering,ierr)

      call MatDestroy(Adj,ierr) 
      call PetscFinalize(ierr)
      end

