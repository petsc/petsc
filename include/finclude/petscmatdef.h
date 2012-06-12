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
#define MatCoarsen PetscFortranAddr
#define MatAIJIndices PetscFortranAddr
#define MatType character*(80)
#define MatSolverPackage character*(80)
#define MatOption PetscEnum
#define MatGetSubMatrixOption PetscEnum
#define MPChacoGlobalType PetscEnum
#define MPChacoLocalType PetscEnum
#define MPChacoEigenType PetscEnum
#define MPPTScotchStragegyType PetscEnum
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
#define MatCoarsenType character*(80)
#define MatCompositeType PetscEnum
#define MatStencil PetscInt
#define MatStencil_k 1
#define MatStencil_j 2
#define MatStencil_i 3
#define MatStencil_c 4

#define MATPARTITIONING_CURRENT 'current'
#define MATPARTITIONING_PARMETIS 'parmetis'

#define MATCOARSEN_MIS 'mis'

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
!
!  Matrix types
!
#define MATSAME            'same'
#define MATMAIJ            'maij'
#define MATSEQMAIJ         'seqmaij'
#define MATMPIMAIJ         'mpimaij'
#define MATIS              'is'
#define MATAIJ             'aij'
#define MATSEQAIJ          'seqaij'
#define MATMPIAIJ          'mpiaij'
#define MATAIJCRL          'aijcrl'
#define MATSEQAIJCRL       'seqaijcrl'
#define MATMPIAIJCRL       'mpiaijcrl'
#define MATAIJCUSP         'aijcusp'
#define MATSEQAIJCUSP      'seqaijcusp'
#define MATMPIAIJCUSP      'mpiaijcusp'
#define MATAIJPERM         'aijperm'
#define MATSEQAIJPERM      'seqaijperm'
#define MATMPIAIJPERM      'mpiaijperm'
#define MATSHELL           'shell'
#define MATDENSE           'dense'
#define MATSEQDENSE        'seqdense'
#define MATMPIDENSE        'mpidense'
#define MATBAIJ            'baij'
#define MATSEQBAIJ         'seqbaij'
#define MATMPIBAIJ         'mpibaij'
#define MATMPIADJ          'mpiadj'
#define MATSBAIJ           'sbaij'
#define MATSEQSBAIJ        'seqsbaij'
#define MATMPISBAIJ        'mpisbaij'

#define MATSEQBSTRM        'seqbstrm'
#define MATMPIBSTRM        'mpibstrm'
#define MATBSTRM           'bstrm'
#define MATSEQSBSTRM       'seqsbstrm'
#define MATMPISBSTRM       'mpisbstrm'
#define MATSBSTRM          'sbstrm'

#define MATDAAD            'daad'
#define MATMFFD            'mffd'
#define MATNORMAL          'normal'
#define MATLRC             'lrc'
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
#define MATLOCALREF        'localref'
#define MATNEST            'nest'
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
#define MATSOLVERMATLAB       'matlab'
#define MATSOLVERPETSC        'petsc'
#define MATSOLVERPLAPACK      'plapack'
#define MATSOLVERBAS          'bas'

#define MATSOLVERBSTRM        'bstrm'
#define MATSOLVERSBSTRM       'sbstrm'
#endif
