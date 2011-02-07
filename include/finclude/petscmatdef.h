!
!
!  Include file for Fortran use of the Mat package in PETSc
!
#if !defined (__PETSCMATDEF_H)
#define __PETSCMATDEF_H

#include "finclude/petscvecdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define Mat PetscFortranAddr
#define MatNullSpace PetscFortranAddr
#define MatFDColoring PetscFortranAddr
#endif
#define MatPartitioning PetscFortranAddr
#define MatAIJIndices PetscFortranAddr
#define MatType character*(80)
#define MatSolverPackage character*(80)
#define MatOption PetscEnum
#define MatGetSubMatrixOption PetscEnum
#define MPChacoGlobalType PetscEnum
#define MPChacoLocalType PetscEnum
#define MPChacoEigenType PetscEnum
#define MPScotchGlobalType PetscEnum
#define MPScotchLocalType PetscEnum
#define MatAssemblyType PetscEnum
#define MatFactorType PetscEnum
#define MatFactorShiftType PetscEnum
#define MatOrderingType character*(80)
#define MatSORType PetscEnum
#define MatInfoType PetscEnum
#define MatReuse PetscEnum
#define MatOperation PetscEnum
#define MatColoringType character*(80)
#define MatInfo PetscLogDouble
#define MatFactorInfo PetscReal
#define MatDuplicateOption PetscEnum
#define MatStructure PetscEnum
#define MatPartitioningType character*(80)
#define MatCompositeType PetscEnum
#define MatStencil PetscInt
#define MatStencil_k 1
#define MatStencil_j 2
#define MatStencil_i 3
#define MatStencil_c 4

#define MATPARTITIONING_CURRENT 'current'
#define MATPARTITIONING_PARMETIS 'parmetis'

#define MATCOLORINGNATURAL 'natural'
#define MATCOLORINGSL 'sl'
#define MATCOLORINGLF 'lf'
#define MATCOLORINGID 'id'

#define MATORDERINGNATURAL 'natural'
#define MATORDERINGND 'nd'
#define MATORDERING1WD '1wd'
#define MATORDERINGRCM 'rcm'
#define MATORDERINGQMD 'qmd'
#define MATORDERINGROWLENGTH 'rowlength'
#define MATORDERINGDSC_ND 'dsc_nd'
#define MATORDERINGDSC_MMD 'dsc_mmd'
#define MATORDERINGDSC_MDF 'dsc_mdf'

!
!  Matrix types
!
#define MATSAME            'same'
#define MATSEQMAIJ         'seqmaij'
#define MATMPIMAIJ         'mpimaij'
#define MATMAIJ            'maij'
#define MATIS              'is'
#define MATSEQAIJ          'seqaij'
#define MATMPIAIJ          'mpiaij'
#define MATAIJ             'aij'
#define MATSHELL           'shell'
#define MATSEQDENSE        'seqdense'
#define MATMPIDENSE        'mpidense'
#define MATDENSE           'dense'
#define MATSEQBAIJ         'seqbaij'
#define MATMPIBAIJ         'mpibaij'
#define MATBAIJ            'baij'
#define MATMPIADJ          'mpiadj'
#define MATSEQSBAIJ        'seqsbaij'
#define MATMPISBAIJ        'mpisbaij'
#define MATSBAIJ           'sbaij'
#define MATDAAD            'daad'
#define MATMFFD            'mffd'
#define MATNORMAL          'normal'
#define MATLRC             'lrc'
#define MATSEQAIJPERM      'seqaijperm'
#define MATMPIAIJPERM      'mpiaijperm'
#define MATAIJPERM         'aijperm'
#define MATSEQAIJCRL       'seqaijcrl'
#define MATMPIAIJCRL       'mpiaijcrl'
#define MATAIJCRL          'aijcrl'
#define MATSCATTER         'scatter'
#define MATBLOCKMAT        'blockmat'
#define MATCOMPOSITE       'composite'
#define MATFFT             'fft'
#define MATFFTW            'fftw'
#define MATSEQCUFFT        'seqcufft'
#define MATTRANSPOSEMAT    'transpose'
#define MATSCHURCOMPLEMENT 'schurcomplement'
#define MATPYTHON          'python'
#define MATHYPRESTRUCT     'hyprestruct'
#define MATHYPRESSTRUCT    'hypresstruct'
#define MATSUBMATRIX       'submatrix'
#ifdef PETSC_USE_MATFWK
#define MATFWK           'matfwk'
#endif
!
! MatSolverPackages
!
#define MATSOLVERSPOOLES      'spooles'
#define MATSOLVERSUPERLU      'superlu'
#define MATSOLVERSUPERLU_DIST 'superlu_dist'
#define MATSOLVERUMFPACK      'umfpack'
#define MATSOLVERCHOLMOD      'cholmod'
#define MATSOLVERESSL         'essl'
#define MATSOLVERLUSOL        'lusol'
#define MATSOLVERMUMPS        'mumps'
#define MATSOLVERPASTIX       'pastix'
#define MATSOLVERDSCPACK      'dscpack'
#define MATSOLVERMATLAB       'matlab'
#define MATSOLVERPETSC        'petsc'
#define MATSOLVERPLAPACK      'plapack'
#define MATSOLVERBAS          'bas'
#endif
