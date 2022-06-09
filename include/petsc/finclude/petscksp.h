!
!
!  Include file for Fortran use of the KSP package in PETSc
!
#if !defined (PETSCKSPDEF_H)
#define PETSCKSPDEF_H

#include "petsc/finclude/petscpc.h"

#define KSP type(tKSP)
#define KSPGuess type(tKSPGuess)

#define KSPType character*(80)
#define KSPGuessType character*(80)
#define KSPCGType PetscEnum
#define KSPFCDTruncationType PetscEnum
#define KSPConvergedReason PetscEnum
#define KSPNormType PetscEnum
#define KSPGMRESCGSRefinementType PetscEnum
#define MatSchurComplementAinvType PetscEnum
#define MatLMVMSymBroydenScaleType PetscEnum
#define KSPHPDDMType PetscEnum

!
!  Various Krylov subspace methods
!
#define KSPRICHARDSON 'richardson'
#define KSPCHEBYSHEV 'chebyshev'
#define KSPCG 'cg'
#define KSPGROPPCG 'groppcg'
#define KSPPIPECG 'pipecg'
#define KSPPIPECGRR 'pipecgrr'
#define KSPPIPELCG 'pipelcg'
#define KSPPIPECG2 'pipecg2'
#define KSPCGNE 'cgne'
#define KSPNASH 'nash'
#define KSPSTCG 'stcg'
#define KSPGLTR 'gltr'
#define KSPFCG 'fcg'
#define KSPPIPEFCG 'pipefcg'
#define KSPGMRES 'gmres'
#define KSPPIPEFGMRES 'pipefgmres'
#define KSPFGMRES 'fgmres'
#define KSPLGMRES 'lgmres'
#define KSPDGMRES 'dgmres'
#define KSPPGMRES 'pgmres'
#define KSPTCQMR 'tcqmr'
#define KSPBCGS 'bcgs'
#define KSPIBCGS 'ibcgs'
#define KSPQMRCGS 'qmrcgs'
#define KSPFBCGS  'fbcgs'
#define KSPFBCGSR 'fbcgsr'
#define KSPBCGSL 'bcgsl'
#define KSPPIPEBCGS 'pipebcgs'
#define KSPCGS 'cgs'
#define KSPTFQMR 'tfqmr'
#define KSPCR 'cr'
#define KSPPIPECR 'pipecr'
#define KSPLSQR 'lsqr'
#define KSPPREONLY 'preonly'
#define KSPNONE 'none'
#define KSPQCG 'qcg'
#define KSPBICG 'bicg'
#define KSPMINRES 'minres'
#define KSPSYMMLQ 'symmlq'
#define KSPLCD 'lcd'
#define KSPPYTHON 'python'
#define KSPGCR 'gcr'
#define KSPPIPEGCR 'pipegcr'
#define KSPTSIRM 'tsirm'
#define KSPCGLS 'cgls'
#define KSPFETIDP 'fetidp'
#define KSPHPDDM 'hpddm'
!
!  Various Initial guesses for Krylov subspace methods
!
#define KSPGUESSFISCHER 'fischer'
#define KSPGUESSPOD 'pod'
#endif
