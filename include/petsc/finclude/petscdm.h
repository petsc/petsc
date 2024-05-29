!
!  Include file for Fortran use of the DM package in PETSc
!
#if !defined (PETSCDMDEF_H)
#define PETSCDMDEF_H

#include "petsc/finclude/petscis.h"
#include "petsc/finclude/petscvec.h"
#include "petsc/finclude/petscmat.h"
#include "petsc/finclude/petscdt.h"

#define DMType character*(80)
#define DMBoundaryType               PetscEnum
#define DMBlockingType               PetscEnum
#define DMPointLocationType          PetscEnum
#define DMAdaptationType             PetscEnum
#define DMAdaptFlag                  PetscEnum
#define PetscUnit                    PetscEnum
#define DMAdaptationStrategy         PetscEnum
#define DMDirection                  PetscEnum
#define DMEnclosureType              PetscEnum
#define DMPolytopeType               PetscEnum
#define DMCopyLabelsMode             PetscEnum
#define PetscDTSimplexQuadratureType PetscEnum
#define DMReorderDefaultFlag         PetscEnum

#define DM               type(tDM)
#define DMAdaptor        type(tDMAdaptor)
#define PetscQuadrature  type(tPetscQuadrature)
#define PetscWeakForm    type(tPetscWeakForm)
#define PetscDS          type(tPetscDS)
#define PetscFE          type(tPetscFE)
#define PetscSpace       type(tPetscSpace)
#define PetscDualSpace   type(tPetscDualSpace)
#define PetscFV          type(tPetscFV)
#define PetscLimiter     type(tPetscLimiter)
#define PetscPartitioner type(tPetscPartitioner)
#define DMField          type(tDMField)

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
#define DMPLEXREFINE1D            'refine_1d'
#define DMPLEXEXTRUDE             'extrude'
#define DMPLEXTRANSFORMFILTER     'transform_filter'

#endif
