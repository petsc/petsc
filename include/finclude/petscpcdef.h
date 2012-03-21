!
!
!  Include file for Fortran use of the PC (preconditioner) package in PETSc
!
#if !defined (__PETSCPCDEF_H)
#define __PETSCPCDEF_H

#include "finclude/petscmatdef.h"
#include "finclude/petscdmdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define PC PetscFortranAddr
#endif
#define PCSide PetscEnum
#define PCASMType PetscEnum
#define PCCompositeType PetscEnum
#define PCRichardsonConvergedReason PetscEnum 
#define PCType character*(80)
#define PCFieldSplitSchurPreType PetscEnum
#define PCPARMSGlobalType PetscEnum
#define PCPARMSLocalType PetscEnum
#define PCFieldSplitSchurFactType PetscEnum
#define CoarseProblemType PetscEnum
#define PCGAMGType PetscEnum
!
!  Various preconditioners
!
#define PCNONE 'none'
#define PCJACOBI 'jacobi'
#define PCSOR 'sor'
#define PCLU 'lu'
#define PCSHELL 'shell'
#define PCBJACOBI 'bjacobi'
#define PCMG 'mg'
#define PCEISENSTAT 'eisenstat'
#define PCILU 'ilu'
#define PCICC 'icc'
#define PCASM 'asm'
#define PCGASM 'gasm'
#define PCKSP 'ksp'
#define PCCOMPOSITE 'composite'
#define PCREDUNDANT 'redundant'
#define PCSPAI 'spai'
#define PCNN 'nn'
#define PCCHOLESKY 'cholesky'
#define PCPBJACOBI 'pbjacobi'
#define PCMAT 'mat'
#define PCHYPRE 'hypre'
#define PCPARMS 'parms'
#define PCFIELDSPLIT 'fieldsplit'
#define PCTFS 'tfs'
#define PCML 'ml'
#define PCPROMETHEUS 'prometheus'
#define PCGALERKIN 'galerkin'
#define PCEXOTIC 'exotic'
#define PCHMPI 'hmpi'
#define PCSUPPORTGRAPH 'supportgraph'
#define PCASA 'asa'
#define PCCP 'cp'
#define PCBFBT 'bfbt'
#define PCLSC 'lsc'
#define PCPYTHON 'python'
#define PCPFMG 'pfmg'
#define PCSYSPFMG 'syspfmg'
#define PCREDISTRIBUTE 'redistribute'
#define PCSVD 'svd'
#define PCGAMG 'gamg'
#define PCSACUSP 'sacusp'
#define PCSACUSPPOLY 'sacusppoly'
#define PCBICGSTABCUSP 'bicgstabcusp'
#define PCAINVCUSP 'ainvcusp'
#define PCBDDC 'bddc'

#endif
