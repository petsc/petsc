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

#define MATCOLORING_NATURAL 'natural'
#define MATCOLORING_SL 'sl'
#define MATCOLORING_LF 'lf'
#define MATCOLORING_ID 'id'

#define MATORDERING_NATURAL 'natural'
#define MATORDERING_ND 'nd'
#define MATORDERING_1WD '1wd'
#define MATORDERING_RCM 'rcm'
#define MATORDERING_QMD 'qmd'
#define MATORDERING_ROWLENGTH 'rowlength'
#define MATORDERING_DSC_ND 'dsc_nd'
#define MATORDERING_DSC_MMD 'dsc_mmd'
#define MATORDERING_DSC_MDF 'dsc_mdf'

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
#define MATSEQCSRPERM      'seqcsrperm'
#define MATMPICSRPERM      'mpicsrperm'
#define MATCSRPERM         'csrperm'
#define MATSEQCRL          'seqcrl'
#define MATMPICRL          'mpicrl'
#define MATCRL             'crl'
#define MATSCATTER         'scatter'
#define MATBLOCKMAT        'blockmat'
#define MATCOMPOSITE       'composite'
#define MATSEQFFTW         'seqfftw'
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
#define MAT_SOLVER_SPOOLES      'spooles'
#define MAT_SOLVER_SUPERLU      'superlu'
#define MAT_SOLVER_SUPERLU_DIST 'superlu_dist'
#define MAT_SOLVER_UMFPACK      'umfpack'
#define MAT_SOLVER_ESSL         'essl'
#define MAT_SOLVER_LUSOL        'lusol'
#define MAT_SOLVER_MUMPS        'mumps'
#define MAT_SOLVER_PASTIX       'pastix'
#define MAT_SOLVER_DSCPACK      'dscpack'
#define MAT_SOLVER_MATLAB       'matlab'
#define MAT_SOLVER_PETSC        'petsc'
#define MAT_SOLVER_PLAPACK      'plapack'
#define MAT_SOLVER_BAS          'bas'
#endif
