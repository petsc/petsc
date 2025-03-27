#pragma once

/* MANSEC = KSP */
/* SUBMANSEC = PC */

/*S
   PC - Abstract PETSc object that manages all preconditioners including direct solvers such as `PCLU`

   Level: beginner

.seealso: [](doc_linsolve), [](sec_pc), `PCCreate()`, `PCSetType()`, `PCType`
S*/
typedef struct _p_PC *PC;

/*J
   PCType - String with the name of a PETSc preconditioner.  These are all the preconditioners and direct solvers that PETSc provides.

   Level: beginner

   Notes:
   Use `PCSetType()` or the options database key `-pc_type` to set the preconditioner to use with a given `PC` object

   `PCRegister()` is used to register preconditioners that are then accessible via `PCSetType()`

.seealso: [](doc_linsolve), [](sec_pc), `PCSetType()`, `PC`, `PCCreate()`, `PCRegister()`, `PCSetFromOptions()`, `PCLU`, `PCJACOBI`, `PCBJACOBI`
J*/
typedef const char *PCType;
#define PCNONE               "none"
#define PCJACOBI             "jacobi"
#define PCSOR                "sor"
#define PCLU                 "lu"
#define PCQR                 "qr"
#define PCSHELL              "shell"
#define PCAMGX               "amgx"
#define PCBJACOBI            "bjacobi"
#define PCMG                 "mg"
#define PCEISENSTAT          "eisenstat"
#define PCILU                "ilu"
#define PCICC                "icc"
#define PCASM                "asm"
#define PCGASM               "gasm"
#define PCKSP                "ksp"
#define PCBJKOKKOS           "bjkokkos"
#define PCCOMPOSITE          "composite"
#define PCREDUNDANT          "redundant"
#define PCSPAI               "spai"
#define PCNN                 "nn"
#define PCCHOLESKY           "cholesky"
#define PCPBJACOBI           "pbjacobi"
#define PCVPBJACOBI          "vpbjacobi"
#define PCMAT                "mat"
#define PCHYPRE              "hypre"
#define PCPARMS              "parms"
#define PCFIELDSPLIT         "fieldsplit"
#define PCTFS                "tfs"
#define PCML                 "ml"
#define PCGALERKIN           "galerkin"
#define PCEXOTIC             "exotic"
#define PCCP                 "cp"
#define PCLSC                "lsc"
#define PCPYTHON             "python"
#define PCPFMG               "pfmg"
#define PCSMG                "smg"
#define PCSYSPFMG            "syspfmg"
#define PCREDISTRIBUTE       "redistribute"
#define PCSVD                "svd"
#define PCGAMG               "gamg"
#define PCCHOWILUVIENNACL    "chowiluviennacl"
#define PCROWSCALINGVIENNACL "rowscalingviennacl"
#define PCSAVIENNACL         "saviennacl"
#define PCBDDC               "bddc"
#define PCKACZMARZ           "kaczmarz"
#define PCTELESCOPE          "telescope"
#define PCPATCH              "patch"
#define PCLMVM               "lmvm"
#define PCHMG                "hmg"
#define PCDEFLATION          "deflation"
#define PCHPDDM              "hpddm"
#define PCH2OPUS             "h2opus"
#define PCMPI                "mpi"

/*E
    PCSide - Determines if the preconditioner is to be applied to the left, right
             or symmetrically around the operator in `KSPSolve()`.

   Values:
+  `PC_LEFT`      - applied after the operator is applied
.  `PC_RIGHT`     - applied before the operator is applied
-  `PC_SYMMETRIC` - a portion of the preconditioner is applied before the operator and the transpose of this portion is applied after the operator is applied.

   Level: beginner

   Note:
   Certain `KSPType` support only a subset of `PCSide` values

.seealso: [](sec_pc), `PC`, `KSPSetPCSide()`, `KSP`, `KSPType`, `KSPGetPCSide()`, `KSPSolve()`
E*/
typedef enum {
  PC_SIDE_DEFAULT = -1,
  PC_LEFT         = 0,
  PC_RIGHT        = 1,
  PC_SYMMETRIC    = 2
} PCSide;
#define PC_SIDE_MAX (PC_SYMMETRIC + 1)

/*E
    PCRichardsonConvergedReason - reason a `PCApplyRichardson()` method terminated

   Level: advanced

.seealso: [](sec_pc), `KSPRICHARDSON`, `PC`, `PCApplyRichardson()`
E*/
typedef enum {
  PCRICHARDSON_NOT_SET        = 0,
  PCRICHARDSON_CONVERGED_RTOL = 2,
  PCRICHARDSON_CONVERGED_ATOL = 3,
  PCRICHARDSON_CONVERGED_ITS  = 4,
  PCRICHARDSON_DIVERGED_DTOL  = -4
} PCRichardsonConvergedReason;

/*E
    PCJacobiType - Determines what elements of the matrix are used to form the Jacobi preconditioner, that is with the `PCType` of `PCJACOBI`

   Values:
+  `PC_JACOBI_DIAGONAL` - use the diagonal entry, if it is zero use one
.  `PC_JACOBI_ROWL1`    - add sum of absolute values in row i, j != i, to diag_ii
.  `PC_JACOBI_ROWMAX`   - use the maximum absolute value in the row
-  `PC_JACOBI_ROWSUM`   - use the sum of the values in the row (not the absolute values)

   Level: intermediate

.seealso: [](sec_pc), `PCJACOBI`, `PC`
E*/
typedef enum {
  PC_JACOBI_DIAGONAL,
  PC_JACOBI_ROWL1,
  PC_JACOBI_ROWMAX,
  PC_JACOBI_ROWSUM
} PCJacobiType;

/*E
    PCASMType - Determines the type of additive Schwarz method, `PCASM`, to use

   Values:
+  `PC_ASM_BASIC`        - Symmetric version where residuals from the ghost points are used
                           and computed values in ghost regions are added together.
                           Classical standard additive Schwarz as introduced in {cite}`dryja1987additive`.
.  `PC_ASM_RESTRICT`     - Residuals from ghost points are used but computed values in ghost
                           region are discarded {cite}`cs99`. Default.
.  `PC_ASM_INTERPOLATE`  - Residuals from ghost points are not used, computed values in ghost
                           region are added back in.
-  `PC_ASM_NONE`         - Residuals from ghost points are not used, computed ghost values are
                           discarded. Not very good.

   Level: beginner

.seealso: [](sec_pc), `PC`, `PCASM`, `PCASMSetType()`, `PCGASMType`
E*/
typedef enum {
  PC_ASM_BASIC       = 3,
  PC_ASM_RESTRICT    = 1,
  PC_ASM_INTERPOLATE = 2,
  PC_ASM_NONE        = 0
} PCASMType;

/*E
    PCGASMType - Determines the type of generalized additive Schwarz method to use (differs from `PCASM` in allowing multiple processors per subdomain) with the `PCType` of `PCGASM`

   Values:
+  `PC_GASM_BASIC`       - Symmetric version where the full from the outer subdomain is used, and the resulting correction is applied
                           over the outer subdomains.  As a result, points in the overlap will receive the sum of the corrections
                           from neighboring subdomains. Classical standard additive Schwarz {cite}`dryja1987additive`.
.  `PC_GASM_RESTRICT`    - Residual from the outer subdomain is used but the correction is restricted to the inner subdomain only
                           (i.e., zeroed out over the overlap portion of the outer subdomain before being applied).  As a result,
                           each point will receive a correction only from the unique inner subdomain containing it (nonoverlapping covering
                           assumption) {cite}`cs99`. Default.
.  `PC_GASM_INTERPOLATE` - Residual is zeroed out over the overlap portion of the outer subdomain, but the resulting correction is
                           applied over the outer subdomain. As a result, points in the overlap will receive the sum of the corrections
                           from neighboring subdomains.
-  `PC_GASM_NONE`        - Residuals and corrections are zeroed out outside the local subdomains. Not very good.

   Level: beginner

   Note:
   Each subdomain has nested inner and outer parts.  The inner subdomains are assumed to form a non-overlapping covering of the computational
   domain, while the outer subdomains contain the inner subdomains and overlap with each other. The `PCGASM` preconditioner will compute
   a subdomain correction over each *outer* subdomain from a residual computed there, but its different variants will differ in
   (a) how the outer subdomain residual is computed, and (b) how the outer subdomain correction is computed.

   Developer Note:
   Perhaps better to remove this since it matches `PCASMType`

.seealso: [](sec_pc), `PCGASM`, `PCASM`, `PC`, `PCGASMSetType()`, `PCASMType`
E*/
typedef enum {
  PC_GASM_BASIC       = 3,
  PC_GASM_RESTRICT    = 1,
  PC_GASM_INTERPOLATE = 2,
  PC_GASM_NONE        = 0
} PCGASMType;

/*E
    PCCompositeType - Determines how two or more preconditioner are composed with the `PCType` of `PCCOMPOSITE`

  Values:
+  `PC_COMPOSITE_ADDITIVE`                 - results from application of all preconditioners are added together
.  `PC_COMPOSITE_MULTIPLICATIVE`           - preconditioners are applied sequentially to the residual freshly
                                             computed after the previous preconditioner application
.  `PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE` - preconditioners are applied sequentially to the residual freshly
                                             computed from first preconditioner to last and then back (Use only for symmetric matrices and preconditioners)
.  `PC_COMPOSITE_SPECIAL`                  - This is very special for a matrix of the form $ \alpha I + R + S$
                                             where the first preconditioner is built from $\alpha I + S$ and second from $\alpha I + R$
.  `PC_COMPOSITE_SCHUR`                    - composes the Schur complement of the matrix from two blocks, see `PCFIELDSPLIT`
-  `PC_COMPOSITE_GKB`                      - the generalized Golub-Kahan bidiagonalization preconditioner, see `PCFIELDSPLIT`

   Level: beginner

.seealso: [](sec_pc), `PCCOMPOSITE`, `PCFIELDSPLIT`, `PC`, `PCCompositeSetType()`, `SNESCompositeType`, `PCCompositeSpecialSetAlpha()`
E*/
typedef enum {
  PC_COMPOSITE_ADDITIVE,
  PC_COMPOSITE_MULTIPLICATIVE,
  PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE,
  PC_COMPOSITE_SPECIAL,
  PC_COMPOSITE_SCHUR,
  PC_COMPOSITE_GKB
} PCCompositeType;

/*E
    PCFieldSplitSchurPreType - Determines how to precondition a Schur complement arising with the `PCType` of `PCFIELDSPLIT`

    Values:
+  `PC_FIELDSPLIT_SCHUR_PRE_SELF`  - the preconditioner for the Schur complement is generated from the symbolic representation of the Schur complement matrix.
                                     The only preconditioners that currently work with this symbolic representation matrix object are `PCLSC` and `PCHPDDM`
.  `PC_FIELDSPLIT_SCHUR_PRE_SELFP` - the preconditioning for the Schur complement is generated from an explicitly-assembled approximation $Sp = A11 - A10 diag(A00)^{-1} A01$.
                                     This is only a good preconditioner when $diag(A00)$ is a good preconditioner for $A00$. Optionally, $A00$ can be
                                     lumped before extracting the diagonal using the additional option `-fieldsplit_1_mat_schur_complement_ainv_type lump`
.  `PC_FIELDSPLIT_SCHUR_PRE_A11`   - the preconditioner for the Schur complement is generated from $A11$, not the Schur complement matrix
.  `PC_FIELDSPLIT_SCHUR_PRE_USER`  - the preconditioner for the Schur complement is generated from the user provided matrix (pre argument
                                     to this function).
-  `PC_FIELDSPLIT_SCHUR_PRE_FULL`  - the preconditioner for the Schur complement is generated from the exact Schur complement matrix representation
                                     computed internally by `PCFIELDSPLIT` (this is expensive) useful mostly as a test that the Schur complement approach can work for your problem

    Level: intermediate

.seealso: [](sec_pc), `PCFIELDSPLIT`, `PCFieldSplitSetSchurPre()`, `PC`
E*/
typedef enum {
  PC_FIELDSPLIT_SCHUR_PRE_SELF,
  PC_FIELDSPLIT_SCHUR_PRE_SELFP,
  PC_FIELDSPLIT_SCHUR_PRE_A11,
  PC_FIELDSPLIT_SCHUR_PRE_USER,
  PC_FIELDSPLIT_SCHUR_PRE_FULL
} PCFieldSplitSchurPreType;

/*E
    PCFieldSplitSchurFactType - determines which off-diagonal parts of the approximate block factorization to use with the `PCType` of `PCFIELDSPLIT`

    Values:
+   `PC_FIELDSPLIT_SCHUR_FACT_DIAG`  - the preconditioner is solving `D`
.   `PC_FIELDSPLIT_SCHUR_FACT_LOWER` - the preconditioner is solving `L D`
.   `PC_FIELDSPLIT_SCHUR_FACT_UPPER` - the preconditioner is solving `D U`
-   `PC_FIELDSPLIT_SCHUR_FACT_FULL`  - the preconditioner is solving `L(D U)`

    where the matrix is factorized as
.vb
   (A   B)  = (1       0) (A   0) (1  Ainv*B)  = L D U
   (C   E)    (C*Ainv  1) (0   S) (0       1)
.ve

    Level: intermediate

.seealso: [](sec_pc), `PCFIELDSPLIT`, `PCFieldSplitSetSchurFactType()`, `PC`
E*/
typedef enum {
  PC_FIELDSPLIT_SCHUR_FACT_DIAG,
  PC_FIELDSPLIT_SCHUR_FACT_LOWER,
  PC_FIELDSPLIT_SCHUR_FACT_UPPER,
  PC_FIELDSPLIT_SCHUR_FACT_FULL
} PCFieldSplitSchurFactType;

/*E
    PCPARMSGlobalType - Determines the global preconditioner method in `PCPARMS`

    Level: intermediate

.seealso: [](sec_pc), `PCPARMS`, `PCPARMSSetGlobal()`, `PC`
E*/
typedef enum {
  PC_PARMS_GLOBAL_RAS,
  PC_PARMS_GLOBAL_SCHUR,
  PC_PARMS_GLOBAL_BJ
} PCPARMSGlobalType;

/*E
    PCPARMSLocalType - Determines the local preconditioner method in `PCPARMS`

    Level: intermediate

.seealso: [](sec_pc), `PCPARMS`, `PCPARMSSetLocal()`, `PC`
E*/
typedef enum {
  PC_PARMS_LOCAL_ILU0,
  PC_PARMS_LOCAL_ILUK,
  PC_PARMS_LOCAL_ILUT,
  PC_PARMS_LOCAL_ARMS
} PCPARMSLocalType;

/*J
    PCGAMGType - type of generalized algebraic multigrid `PCGAMG` method

   Values:
+   `PCGAMGAGG`       - (the default) smoothed aggregation algorithm, robust, very well tested
.   `PCGAMGGEO`       - geometric coarsening, uses mesh generator to produce coarser meshes, limited to triangles, not supported, reference implementation (2D)
-   `PCGAMGCLASSICAL` - classical algebraic multigrid preconditioner, incomplete, not supported, reference implementation

     Level: intermediate

.seealso: [](sec_pc), `PCGAMG`, `PCMG`, `PC`, `PCSetType()`, `PCGAMGSetThreshold()`, `PCGAMGSetThreshold()`, `PCGAMGSetReuseInterpolation()`
J*/
typedef const char *PCGAMGType;
#define PCGAMGAGG       "agg"
#define PCGAMGGEO       "geo"
#define PCGAMGCLASSICAL "classical"

typedef const char *PCGAMGClassicalType;
#define PCGAMGCLASSICALDIRECT   "direct"
#define PCGAMGCLASSICALSTANDARD "standard"

/*E
   PCMGType - Determines the type of multigrid method that is run with the `PCType` of `PCMG`

   Values:
+  `PC_MG_MULTIPLICATIVE` (default) - traditional V or W cycle as determined by `PCMGSetCycleType()`
.  `PC_MG_ADDITIVE`                 - the additive multigrid preconditioner where all levels are
                                      smoothed before updating the residual. This only uses the
                                      down smoother, in the preconditioner the upper smoother is ignored
.  `PC_MG_FULL`                     - same as multiplicative except one also performs grid sequencing,
                                      that is starts on the coarsest grid, performs a cycle, interpolates
                                      to the next, performs a cycle etc. This is much like the F-cycle presented in "Multigrid" by Trottenberg, Oosterlee, Schuller page 49, but that
                                      algorithm supports smoothing on before the restriction on each level in the initial restriction to the coarsest stage. In addition that algorithm
                                      calls the V-cycle only on the coarser level and has a post-smoother instead.
-  `PC_MG_KASKADE`                  - Cascadic or Kaskadic multigrid, like full multigrid except one never goes back to a coarser level from a finer

   Level: beginner

.seealso: [](sec_pc), `PCMG`, `PC`, `PCMGSetType()`, `PCMGSetCycleType()`, `PCMGSetCycleTypeOnLevel()`
E*/
typedef enum {
  PC_MG_MULTIPLICATIVE,
  PC_MG_ADDITIVE,
  PC_MG_FULL,
  PC_MG_KASKADE
} PCMGType;
#define PC_MG_CASCADE PC_MG_KASKADE;

/*E
   PCMGCycleType - Determines which of V-cycle or W-cycle to use with the `PCType` of `PCMG` or `PCGAMG`

   Values:
+  `PC_MG_V_CYCLE` - use the V cycle
-  `PC_MG_W_CYCLE` - use the W cycle

   Level: beginner

.seealso: [](sec_pc), `PCMG`, `PC`, `PCMGSetCycleType()`
E*/
typedef enum {
  PC_MG_CYCLE_V = 1,
  PC_MG_CYCLE_W = 2
} PCMGCycleType;

/*E
    PCMGalerkinType - Determines if the coarse grid operators are computed via the Galerkin process with the `PCType` of `PCMG`

   Values:
+  `PC_MG_GALERKIN_PMAT` - computes the `pmat` (matrix from which the preconditioner is built) via the Galerkin process from the finest grid
.  `PC_MG_GALERKIN_MAT` -  computes the `mat` (matrix used to apply the operator) via the Galerkin process from the finest grid
.  `PC_MG_GALERKIN_BOTH` - computes both the `mat` and `pmat` via the Galerkin process (if pmat == mat the construction is only done once
-  `PC_MG_GALERKIN_NONE` - neither operator is computed via the Galerkin process, the user must provide the operator

   Level: beginner

   Note:
   Users should never set `PC_MG_GALERKIN_EXTERNAL`, it is used by `PCHYPRE` and `PCML`

.seealso: [](sec_pc), `PCMG`, `PC`, `PCMGSetCycleType()`
E*/
typedef enum {
  PC_MG_GALERKIN_BOTH,
  PC_MG_GALERKIN_PMAT,
  PC_MG_GALERKIN_MAT,
  PC_MG_GALERKIN_NONE,
  PC_MG_GALERKIN_EXTERNAL
} PCMGGalerkinType;

/*E
    PCExoticType - Determines which of the face-based or wirebasket-based coarse grid space to use with the `PCType` of `PCEXOTIC`

   Level: beginner

.seealso: [](sec_pc), `PCExoticSetType()`, `PCEXOTIC`
E*/
typedef enum {
  PC_EXOTIC_FACE,
  PC_EXOTIC_WIREBASKET
} PCExoticType;

/*E
   PCBDDCInterfaceExtType - Defines how interface balancing is extended into the interior of subdomains with the `PCType` of `PCBDDC`

   Values:
+  `PC_BDDC_INTERFACE_EXT_DIRICHLET` - solves Dirichlet interior problem; this is the standard BDDC algorithm
-  `PC_BDDC_INTERFACE_EXT_LUMP`      - skips interior solve; sometimes called $M_1$ and associated with "lumped FETI-DP"

   Level: intermediate

.seealso: [](sec_pc), `PCBDDC`, `PC`
E*/
typedef enum {
  PC_BDDC_INTERFACE_EXT_DIRICHLET,
  PC_BDDC_INTERFACE_EXT_LUMP
} PCBDDCInterfaceExtType;

/*E
  PCMGCoarseSpaceType - Function space for coarse space for adaptive interpolation

  Level: beginner

.seealso: [](sec_pc), `PCMGSetAdaptCoarseSpaceType()`, `PCMG`, `PC`
E*/
typedef enum {
  PCMG_ADAPT_NONE,
  PCMG_ADAPT_POLYNOMIAL,
  PCMG_ADAPT_HARMONIC,
  PCMG_ADAPT_EIGENVECTOR,
  PCMG_ADAPT_GENERALIZED_EIGENVECTOR,
  PCMG_ADAPT_GDSW
} PCMGCoarseSpaceType;

/*E
    PCPatchConstructType - Determines the algorithm used to construct patches for the `PCPATCH` preconditioner

   Level: beginner

.seealso: [](sec_pc), `PCPatchSetConstructType()`, `PCPATCH`, `PC`
E*/
typedef enum {
  PC_PATCH_STAR,
  PC_PATCH_VANKA,
  PC_PATCH_PARDECOMP,
  PC_PATCH_USER,
  PC_PATCH_PYTHON
} PCPatchConstructType;

/*E
    PCDeflationSpaceType - Type of deflation used by `PCType` `PCDEFLATION`

    Values:
+   `PC_DEFLATION_SPACE_HAAR`        - directly assembled based on Haar (db2) wavelet with overflowed filter cuted-off
.   `PC_DEFLATION_SPACE_DB2`         - `MATCOMPOSITE` of 1-lvl matices based on db2 (2 coefficient Daubechies / Haar wavelet)
.   `PC_DEFLATION_SPACE_DB4`         - same as above, but with db4 (4 coefficient Daubechies)
.   `PC_DEFLATION_SPACE_DB8`         - same as above, but with db8 (8 coefficient Daubechies)
.   `PC_DEFLATION_SPACE_DB16`        - same as above, but with db16 (16 coefficient Daubechies)
.   `PC_DEFLATION_SPACE_BIORTH22`    - same as above, but with biorthogonal 2.2 (6 coefficients)
.   `PC_DEFLATION_SPACE_MEYER`       - same as above, but with Meyer/FIR (62 coefficients)
.   `PC_DEFLATION_SPACE_AGGREGATION` - aggregates local indices (given by operator matrix distribution) into a subdomain
-   `PC_DEFLATION_SPACE_USER`        - indicates space set by user

    Level: intermediate

    Note:
    Wavelet-based space (except Haar) can be used in multilevel deflation.

.seealso: [](sec_pc), `PCDeflationSetSpaceToCompute()`, `PCDEFLATION`, `PC`
E*/
typedef enum {
  PC_DEFLATION_SPACE_HAAR,
  PC_DEFLATION_SPACE_DB2,
  PC_DEFLATION_SPACE_DB4,
  PC_DEFLATION_SPACE_DB8,
  PC_DEFLATION_SPACE_DB16,
  PC_DEFLATION_SPACE_BIORTH22,
  PC_DEFLATION_SPACE_MEYER,
  PC_DEFLATION_SPACE_AGGREGATION,
  PC_DEFLATION_SPACE_USER
} PCDeflationSpaceType;

/*E
    PCHPDDMCoarseCorrectionType - Type of coarse correction used by `PCType` `PCHPDDM`

    Values:
+   `PC_HPDDM_COARSE_CORRECTION_DEFLATED` (default) - eq. (1) in `PCHPDDMShellApply()`
.   `PC_HPDDM_COARSE_CORRECTION_ADDITIVE`           - eq. (2)
.   `PC_HPDDM_COARSE_CORRECTION_BALANCED`           - eq. (3)
-   `PC_HPDDM_COARSE_CORRECTION_NONE`               - no coarse correction (mostly useful for debugging)

    Level: intermediate

.seealso: [](sec_pc), `PCHPDDM`, `PC`, `PCSetType()`, `PCHPDDMShellApply()`
E*/
typedef enum {
  PC_HPDDM_COARSE_CORRECTION_DEFLATED,
  PC_HPDDM_COARSE_CORRECTION_ADDITIVE,
  PC_HPDDM_COARSE_CORRECTION_BALANCED,
  PC_HPDDM_COARSE_CORRECTION_NONE
} PCHPDDMCoarseCorrectionType;

/*E
    PCHPDDMSchurPreType - Type of `PCHPDDM` preconditioner for a `MATSCHURCOMPLEMENT` generated by `PCFIELDSPLIT` with `PCFieldSplitSchurPreType` set to `PC_FIELDSPLIT_SCHUR_PRE_SELF`

    Values:
+   `PC_HPDDM_SCHUR_PRE_LEAST_SQUARES` (default) - only with a near-zero A11 block and A10 = A01^T; a preconditioner for solving A01^T A00^-1 A01 x = b
                                                   is built by approximating the Schur complement with (inv(sqrt(diag(A00))) A01)^T (inv(sqrt(diag(A00))) A01)
                                                   and by considering the associated linear least squares problem
-   `PC_HPDDM_SCHUR_PRE_GENEO`                   - only with A10 = A01^T, `PCHPDDMSetAuxiliaryMat()` called on the `PC` of the A00 block, and if A11 is nonzero,
                                                   then `PCHPDDMSetAuxiliaryMat()` must be called on the associated `PC` as well (it is built automatically for the
                                                   user otherwise); the Schur complement `PC` is set internally to `PCKSP`, with the prefix `-fieldsplit_1_pc_hpddm_`;
                                                   the operator associated to the `PC` is spectrally equivalent to the original Schur complement

    Level: advanced

.seealso: [](sec_pc), `PCHPDDM`, `PC`, `PCFIELDSPLIT`, `PC_FIELDSPLIT_SCHUR_PRE_SELF`, `PCFieldSplitSetSchurPre()`, `PCHPDDMSetAuxiliaryMat()`
E*/
typedef enum {
  PC_HPDDM_SCHUR_PRE_LEAST_SQUARES,
  PC_HPDDM_SCHUR_PRE_GENEO
} PCHPDDMSchurPreType;

/*E
    PCFailedReason - indicates the type of `PC` failure. That is why the construction of the preconditioner, `PCSetUp()`, or its use, `PCApply()`, failed

    Level: beginner

.seealso: [](sec_pc), `PC`, `PCGetFailedReason()`, `PCSetUp()`
E*/
typedef enum {
  PC_SETUP_ERROR              = -1,
  PC_NOERROR                  = 0,
  PC_FACTOR_STRUCT_ZEROPIVOT  = 1,
  PC_FACTOR_NUMERIC_ZEROPIVOT = 2,
  PC_FACTOR_OUTMEMORY         = 3,
  PC_FACTOR_OTHER             = 4,
  PC_INCONSISTENT_RHS         = 5,
  PC_SUBPC_ERROR              = 6
} PCFailedReason;

/*E
    PCGAMGLayoutType - Layout for reduced grids for `PCType` `PCGAMG`

    Level: intermediate

.seealso: [](sec_pc), `PCGAMG`, `PC`, `PCGAMGSetCoarseGridLayoutType()`
E*/
typedef enum {
  PCGAMG_LAYOUT_COMPACT,
  PCGAMG_LAYOUT_SPREAD
} PCGAMGLayoutType;
