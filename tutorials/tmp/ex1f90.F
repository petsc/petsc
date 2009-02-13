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
      PetscViewer hdf5viewer
      MatPartitioning mp
      IS partitioning,newnumbering
      IS oldnumbering
      IS all_oldnumbering
      IS allnewnumbering
      IS newnumberingsmall
      PetscInt nlocal,nlocalnew,bs
      PetscInt ntotal
      PetscMPIInt size,rank
      Vec volumes,volumesnew
      VecScatter vs

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      bs = 3
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'mesh.petsc',                &
     &                           FILE_MODE_READ,petscviewer,ierr)

      call MatLoad(petscviewer,MATMPIBAIJ,Adj,ierr)
      call MatGetLocalSize(Adj,nlocal,nlocal,ierr)
      call PetscViewerDestroy(petscviewer,ierr)

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
      call MatGetLocalSize(Adjnew,nlocalnew,nlocalnew,ierr)
      call ISDestroy(allnewnumbering,ierr)

      call PetscViewerHDF5Open(PETSC_COMM_WORLD,'mesh.h5',                &
     &                           FILE_MODE_READ,hdf5viewer,ierr)
      call VecCreateMPI(PETSC_COMM_WORLD,nlocal/bs,PETSC_DETERMINE,       &
     &                  volumes,ierr)
      call PetscObjectSetName(volumes,'/Cells/Volumes',ierr)
      call VecLoadIntoVector(hdf5viewer,volumes,ierr)
      call VecCreateMPI(PETSC_COMM_WORLD,nlocalnew/bs,PETSC_DETERMINE,       &
     &                  volumesnew,ierr)
      call MatGetSize(Adjnew,ntotal,ntotal,ierr)
      call ISCompressIndicesGeneral(ntotal,bs,1, newnumbering,
     &                              newnumberingsmall,ierr)
      call VecScatterCreate(volumes,newnumberingsmall,volumesnew,          &
     &                      PETSC_NULL_OBJECT,vs,ierr)
      call VecScatterBegin(vs,volumes,volumesnew,INSERT_VALUES,           &
     &                     SCATTER_FORWARD,ierr)
      call VecScatterEnd(vs,volumes,volumesnew,INSERT_VALUES,           &
     &                     SCATTER_FORWARD,ierr)

      call PetscViewerDestroy(hdf5viewer,ierr)

      call VecDestroy(volumes,ierr)
      call MatDestroy(Adj,ierr) 
      call MatDestroy(Adjnew,ierr)
      call ISDestroy(newnumbering,ierr)
      call PetscFinalize(ierr)
      end

