! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!             Include file for program ex5f.F
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  This program uses CPP for preprocessing, as indicated by the use of
!  PETSc include files in the directory petsc/include/finclude.  This
!  convention enables use of the CPP preprocessor, which allows the use
!  of the #include statements that define PETSc objects and variables.
!
!  Use of the conventional Fortran include statements is also supported
!  In this case, the PETsc include files are located in the directory
!  petsc/include/foldinclude.
!         
!  Since one must be very careful to include each file no more than once
!  in a Fortran routine, application programmers must explicitly list
!  each file needed for the various PETSc components within their
!  program (unlike the C/C++ interface).
!
!  See the Fortran section of the PETSc users manual for details.
!
!  The following include statements are generally used in SNES Fortran
!  programs:
!     petscsys.h  - base PETSc routines
!     petscvec.h    - vectors
!     petscmat.h    - matrices
!     petscksp.h    - Krylov subspace methods
!     petscpc.h     - preconditioners
!     petscsnes.h   - SNES interface
!  In addition, we need the following for use of distributed arrays
!     petscdmda.h     - distributed arrays (DMDAs)

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscis.h>
#include <finclude/petscmat.h>
#include <finclude/petscksp.h>
#include <finclude/petscpc.h>
#include <finclude/petscsnes.h>

!  Common blocks:
!  In this example we use common blocks to store data needed by the 
!  application-provided call-back routines, FormJacobian() and
!  FormFunction().  Note that we can store (pointers to)
!  PETSc objects within these common blocks. 
!
!  common /params/ - contains parameters for the global application 
!     mx, my   - global discretization in x- and y-directions
!     lambda   - nonlinearity parameter
!
!  common /pdata/  - contains some parallel data
!     da       - distributed array
!     rank     - processor rank within communicator
!     size     - number of processors
!     xs, ys   - local starting grid indices (no ghost points)
!     xm, ym   - widths of local grid (no ghost points)
!     gxs, gys - local starting grid indices (including ghost points)
!     gxm, gym - widths of local grid (including ghost points)

      DM      da
      PetscInt xs,xe,xm,gxs,gxe,gxm
      PetscInt ys,ye,ym,gys,gye,gym
      PetscInt mx,my
      PetscMPIInt rank,size
      PetscReal lambda

      common /params/ lambda,mx,my
      common /pdata/  xs,xe,xm,gxs,gxe,gxm
      common /pdata/  ys,ye,ym,gys,gye,gym
      common /pdata/  da,rank,size

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
