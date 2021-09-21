
!
!  Include file for Fortran use of the DM package in PETSc
!
#if !defined (PETSCDMDEF_H)
#define PETSCDMDEF_H

#include "petsc/finclude/petscis.h"
#include "petsc/finclude/petscvec.h"
#include "petsc/finclude/petscmat.h"

#define DMType character*(80)
#define DMBoundaryType       PetscEnum
#define DMPointLocationType  PetscEnum
#define DMAdaptationType     PetscEnum
#define DMAdaptFlag          PetscEnum
#define PetscUnit            PetscEnum
#define DMAdaptationStrategy PetscEnum
#define DMDirection          PetscEnum
#define DMEnclosureType      PetscEnum
#define DMPolytopeType       PetscEnum
#define DMCopyLabelsMode     PetscEnum

#define DM               type(tDM)

#define DMAdaptor        PetscFortranAddr
#define PetscQuadrature  PetscFortranAddr
#define PetscWeakForm    PetscFortranAddr
#define PetscDS          PetscFortranAddr
#define PetscFE          PetscFortranAddr
#define PetscSpace       PetscFortranAddr
#define PetscDualSpace   PetscFortranAddr
#define PetscFV          PetscFortranAddr
#define PetscLimiter     PetscFortranAddr
#define PetscPartitioner PetscFortranAddr
#define DMField          PetscFortranAddr

#define DMDA        'da'
#define DMCOMPOSITE 'composite'
#define DMSLICED    'sliced'
#define DMSHELL     'shell'
#define DMPLEX      'plex'
#define DMREDUNDANT 'redundant'
#define DMPATCH     'patch'
#define DMMOAB      'moab'
#define DMNETWORK   'network'
#define DMFOREST    'forest'
#define DMP4EST     'p4est'
#define DMP8EST     'p8est'
#define DMSWARM     'swarm'

#define DMPlexTransform type(tDMPlexTransform)

#define DMPLEXREFINEREGULAR       'refine_regular'
#define DMPLEXREFINEALFELD        'refine_alfeld'
#define DMPLEXREFINEPOWELLSABIN   'refine_powell_sabin'
#define DMPLEXREFINEBOUNDARYLAYER 'refine_boundary_layer'
#define DMPLEXREFINESBR           'refine_sbr'
#define DMPLEXREFINETOBOX         'refine_tobox'
#define DMPLEXREFINETOSIMPLEX     'refine_tosimplex'
#define DMPLEXEXTRUDE             'extrude'
#define DMPLEXTRANSFORMFILTER     'transform_filter'

#endif
