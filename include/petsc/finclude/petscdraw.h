!
!
!  Include file for Fortran use of the PetscDraw package in PETSc
!

#if !defined (PETSCDRAWDEF_H)
#define PETSCDRAWDEF_H

#define PetscDraw type(tPetscDraw)
#define PetscDrawLG type(tPetscDrawLG)
#define PetscDrawAxis type(tPetscDrawAxis)
#define PetscDrawSP type(tPetscDrawSP)
#define PetscDrawHG type(tPetscDrawHG)
#define PetscDrawMesh type(tPetscDrawMesh)
#define PetscDrawBar type(tPetscDrawBar)
#define PetscDrawButton PetscEnum
#define PetscDrawType character*(80)
#define PetscDrawMarkerType PetscEnum

!
!  types of draw context
!
#define PETSC_DRAW_X 'x'
#define PETSC_DRAW_NULL 'null'
#define PETSC_DRAW_PS 'ps'
#define PETSC_DRAW_WIN32 'win32'

#endif
