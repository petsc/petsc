program ex95f90
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
#include "exodusII.inc"

  ! Get the Fortran kind associated with PetscInt and PetscReal so that we can use literal constants.
  PetscInt                             :: dummyPetscInt
  PetscReal                            :: dummyPetscreal
  PetscBool                            :: flg
  integer, parameter                    :: kPI = kind(dummyPetscInt)
  integer, parameter                    :: kPR = kind(dummyPetscReal)
  integer                              :: nNodalVar = 4
  integer                              :: nZonalVar = 3
  integer                              :: i

  PetscErrorCode                       :: ierr
  type(tDM)                            :: dm, pdm
  character(len=PETSC_MAX_PATH_LEN)    :: ifilename, ofilename, IOBuffer
  PetscInt                             :: order = 1
  type(tPetscViewer)                   :: viewer
  character(len=MXNAME), dimension(4) :: nodalVarName = ["U_x  ", &
                                                         "U_y  ", &
                                                         "Alpha", &
                                                         "Beta "]
  character(len=MXNAME), dimension(3) :: zonalVarName = ["Sigma_11", &
                                                         "Sigma_12", &
                                                         "Sigma_22"]
  character(len=MXNAME)              :: varName

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
  if (ierr /= 0) then
    print *, 'Unable to initialize PETSc'
    stop
  end if

  PetscCallA(PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, 'PetscViewer_ExodusII test', 'ex95f90', ierr))
  PetscCallA(PetscOptionsString("-i", "Filename to read", "ex95f90", ifilename, ifilename, flg, ierr))
  PetscCheckA(flg, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, 'missing input file name -i <input file name>')
  PetscCallA(PetscOptionsString("-o", "Filename to write", "ex95f90", ofilename, ofilename, flg, ierr))
  PetscCheckA(flg, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, 'missing output file name -o <output file name>')
  PetscCallA(PetscOptionsEnd(ierr))

  ! Read the mesh in any supported format
  PetscCallA(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, PETSC_NULL_CHARACTER, PETSC_TRUE, dm, ierr))
  PetscCallA(DMPlexDistributeSetDefault(dm, PETSC_FALSE, ierr))
  PetscCallA(DMSetFromOptions(dm, ierr))
  PetscCallA(PetscObjectSetName(dm, "ex95f90", ierr))
  PetscCallA(DMViewFromOptions(dm, PETSC_NULL_OBJECT, '-dm_view', ierr))

  ! enable exodus debugging information
#ifdef PETSC_USE_DEBUG
  PetscCallA(exopts(EXVRBS + EXDEBG, ierr))
#endif

  ! Create the exodus file
  PetscCallA(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, ofilename, FILE_MODE_WRITE, viewer, ierr))

  ! Save the geometry to the file, erasing all previous content
  PetscCallA(PetscViewerExodusIISetOrder(viewer, order, ierr))
  PetscCallA(DMView(dm, viewer, ierr))
  PetscCallA(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCall(PetscViewerFlush(viewer, ierr))

  PetscCallA(DMPlexDistribute(dm, 0_kPI, PETSC_NULL_SF, pdm, ierr))
  if (pdm /= PETSC_NULL_DM) Then
    pdm = dm
  end if

  ! Testing Variable Number
  PetscCallA(PetscViewerExodusIISetZonalVariable(viewer, nZonalVar, ierr))
  nZonalVar = -1
  PetscCallA(PetscViewerExodusIIGetZonalVariable(viewer, nZonalVar, ierr))
  Write (IOBuffer, '("Number of zonal variables:", I2,"\n")') nZonalVar
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD, IOBuffer, ierr))

  PetscCallA(PetscViewerExodusIISetNodalVariable(viewer, nNodalVar, ierr))
  nNodalVar = -1
  PetscCallA(PetscViewerExodusIIGetNodalVariable(viewer, nNodalVar, ierr))
  Write (IOBuffer, '("Number of nodal variables:", I2,"\n")') nNodalVar
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD, IOBuffer, ierr))
  PetscCallA(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD, ierr))

  ! Test of PetscViewerExodusIISet[Nodal/Zonal]VariableName
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD, "Testing PetscViewerExodusIISet[Nodal/Zonal]VariableName\n", ierr))
  do i = 1, nZonalVar
    PetscCallA(PetscViewerExodusIISetZonalVariableName(viewer, i - 1, zonalVarName(i), ierr))
  end do
  do i = 1, nNodalVar
    PetscCallA(PetscViewerExodusIISetNodalVariableName(viewer, i - 1, nodalVarName(i), ierr))
  end do
  PetscCall(PetscViewerFlush(viewer, ierr))
  PetscCallA(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD, ierr))

  do i = 1, nZonalVar
    PetscCallA(PetscViewerExodusIIGetZonalVariableName(viewer, i - 1, varName, ierr))
    Write (IOBuffer, '("   zonal variable:", I2,": ",A,"\n")') i, varName
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD, IOBuffer, ierr))
  end do
  do i = 1, nNodalVar
    PetscCallA(PetscViewerExodusIIGetNodalVariableName(viewer, i - 1, varName, ierr))
    Write (IOBuffer, '("   nodal variable:", I2,": ",A,"\n")') i, varName
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD, IOBuffer, ierr))
  end do
  PetscCallA(PetscViewerDestroy(viewer, ierr))

  ! Test of PetscViewerExodusIIGet[Nodal/Zonal]VariableName
  nZonalVar = -1
  nNodalVar = -1
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD, "\n\nReopenning the output file in Read-only mode\n", ierr))
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD, "Testing PetscViewerExodusIIGet[Nodal/Zonal]VariableName\n", ierr))
  PetscCallA(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, ofilename, FILE_MODE_APPEND, viewer, ierr))
  PetscCallA(PetscViewerExodusIISetOrder(viewer, order, ierr))
  PetscCallA(PetscViewerExodusIIGetZonalVariable(viewer, nZonalVar, ierr))
  PetscCallA(PetscViewerExodusIIGetNodalVariable(viewer, nNodalVar, ierr))

  do i = 1, nZonalVar
    PetscCallA(PetscViewerExodusIIGetZonalVariableName(viewer, i - 1, varName, ierr))
    Write (IOBuffer, '("   zonal variable:", I2,": ",A,"\n")') i, varName
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD, IOBuffer, ierr))
  end do
  do i = 1, nNodalVar
    PetscCallA(PetscViewerExodusIIGetNodalVariableName(viewer, i - 1, varName, ierr))
    Write (IOBuffer, '("   nodal variable:", I2,": ",A,"\n")') i, varName
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD, IOBuffer, ierr))
  end do

  PetscCallA(DMDestroy(dm, ierr))
  PetscCallA(PetscViewerDestroy(viewer, ierr))
  PetscCallA(PetscFinalize(ierr))
end program ex95f90

!/*TEST
!
!  build:
!    requires: exodusii pnetcdf !complex
!  test:
!    suffix: 0
!    nsize: 1
!    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo
!
!TEST*/
