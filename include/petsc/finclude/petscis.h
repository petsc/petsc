!
!
!  Include file for Fortran use of the IS (index set) package in PETSc
!
#if !defined (PETSCISDEF_H)
#define PETSCISDEF_H

#include "petsc/finclude/petscsys.h"
#include "petsc/finclude/petscviewer.h"

#define IS type(tIS)
#define ISColoring type(tISColoring)
#define PetscSection type(tPetscSection)
#define PetscSectionSym type(tPetscSectionSym)

#define PetscSF type(tPetscSF)
#define PetscLayout type(tPetscLayout)

#define ISType character*(80)
#define ISInfo PetscEnum
#define ISInfoType PetscEnum
#define ISGlobalToLocalType character*(80)
#define ISGlobalToLocalMappingMode PetscEnum
#define ISColoringType PetscEnum

#define ISLocalToGlobalMapping type(tISLocalToGlobalMapping)

#define ISColoringValue PETSC_IS_COLORING_VALUE_TYPE_F

#define ISGENERAL 'general'
#define ISSTRIDE 'stride'
#define ISBLOCK 'block'

#define ISGLOBALTOLOCALMAPPINGBASIC 'basic'
#define ISGLOBALTOLOCALMAPPINGHASH  'hash'
#endif
