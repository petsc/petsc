
!
!  Include file for Fortran use of the DM package in PETSc
!
#if !defined (__PETSCDMDEF_H)
#define __PETSCDMDEF_H

#include "petsc/finclude/petscisdef.h"
#include "petsc/finclude/petscvecdef.h"
#include "petsc/finclude/petscmatdef.h"

#define DMType character*(80)
#define DMBoundaryType      PetscEnum
#define DMPointLocationType PetscEnum

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define DM               PetscFortranAddr
#define PetscQuadrature  PetscFortranAddr
#define PetscDS          PetscFortranAddr
#define PetscFE          PetscFortranAddr
#define PetscSpace       PetscFortranAddr
#define PetscDualSpace   PetscFortranAddr
#define PetscFV          PetscFortranAddr
#define PetscLimiter     PetscFortranAddr
#define PetscPartitioner PetscFortranAddr
#endif

#define DMDA        'da'
#define DMCOMPOSITE 'composite'
#define DMSLICED    'sliced'
#define DMSHELL     'shell'
#define DMPLEX      'plex'
#define DMCARTESIAN 'cartesian'
#define DMREDUNDANT 'redundant'
#define DMPATCH     'patch'
#define DMMOAB      'moab'
#define DMNETWORK   'network'
#define DMFOREST    'forest'
#define DMP4EST     'p4est'
#define DMP8EST     'p8est'
#define DMSWARM     'swarm'

#endif
