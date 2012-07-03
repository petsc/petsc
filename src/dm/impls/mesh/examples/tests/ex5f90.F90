Program test1f90
#include "finclude/petscdef.h"
   Use petsc
   Implicit NONE

   Type(DM)                               :: dmBody,dmFS
   PetscBool                              :: inflag
   Character(len=256)                     :: infilename,IOBuffer
   PetscErrorCode                         :: ierr
   PetscInt                               :: version
   Integer                                :: rank


   Call PetscInitialize(PETSC_NULL_CHARACTER,ierr); CHKERRQ(ierr)
   Call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)

   Call PetscOptionsGetString(PETSC_NULL_CHARACTER, '-i',infilename,inflag,ierr)
   CHKERRQ(ierr)
   If (.NOT. inflag) Then
      Call PetscPrintf(PETSC_COMM_WORLD,"No file name given\n",iErr);CHKERRQ(ierr)
      Call PetscFinalize(iErr)
      STOP
   End If

   !!!
   !!!   Reads a mesh
   !!!
   version = 1
   Call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-v',version,inflag,ierr)
   CHKERRQ(ierr)
   Call DMMeshCreateExodusNG(PETSC_COMM_WORLD,infilename,dmBody,dmFS,ierr)
   CHKERRQ(ierr)
   Call PetscPrintf(PETSC_COMM_WORLD,"dmBody:\n",ierr);CHKERRQ(ierr)
   Call DMView(dmBody,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRQ(ierr)
   Call PetscPrintf(PETSC_COMM_WORLD,"dmFS:\n",ierr);CHKERRQ(ierr)
   Call DMView(dmFS,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRQ(ierr)

   Call DMDestroy(dmBody,ierr);CHKERRQ(ierr)
   Call DMDestroy(dmFS,ierr);CHKERRQ(ierr)
   Call PetscFinalize(iErr)
End Program test1f90