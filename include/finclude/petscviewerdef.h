!
!  Include file for Fortran use of the PetscViewer package in PETSc
!
#if !defined (__PETSCVIEWERDEF_H)
#define __PETSCVIEWERDEF_H

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define PetscViewer PetscFortranAddr
#endif

#define PetscViewers PetscFortranAddr
#define PetscFileMode PetscEnum
#define PetscViewerType character*(80)
#define PetscViewerFormat PetscEnum

#define PETSCVIEWERSOCKET 'socket'
#define PETSCVIEWERASCII 'ascii'
#define PETSCVIEWERBINARY 'binary'
#define PETSCVIEWERSTRING 'string'
#define PETSCVIEWERDRAW 'draw'
#define PETSCVIEWERVU 'vu'
#define PETSCVIEWERMATHEMATICA 'mathematica'
#define PETSCVIEWERNETCDF 'netcdf'
#define PETSCVIEWERHDF5 'hdf5'
#define PETSCVIEWERMATLAB 'matlab'
#define PETSCVIEWERAMS 'ams'

#endif
