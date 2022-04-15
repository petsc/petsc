
#include <petscao.h>
#include <petsc/private/dmlabelimpl.h>
#include <petsc/private/dmfieldimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/dmplextransformimpl.h>
#include <petsc/private/petscdsimpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petsc/private/petscfvimpl.h>
#include <petsc/private/dmswarmimpl.h>

static PetscBool DMPackageInitialized = PETSC_FALSE;
/*@C
  DMFinalizePackage - This function finalizes everything in the DM package. It is called
  from PetscFinalize().

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode  DMFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&DMList));
  DMPackageInitialized = PETSC_FALSE;
  DMRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HYPRE)
PETSC_EXTERN PetscErrorCode MatCreate_HYPREStruct(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_HYPRESStruct(Mat);
#endif

/*@C
  DMInitializePackage - This function initializes everything in the DM package. It is called
  from PetscDLLibraryRegister_petscdm() when using dynamic libraries, and on the first call to AOCreate()
  or DMDACreate() when using shared or static libraries.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode DMInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (DMPackageInitialized) PetscFunctionReturn(0);
  DMPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  PetscCall(PetscClassIdRegister("Distributed Mesh",&DM_CLASSID));
  PetscCall(PetscClassIdRegister("DM Label",&DMLABEL_CLASSID));
  PetscCall(PetscClassIdRegister("Quadrature",&PETSCQUADRATURE_CLASSID));
  PetscCall(PetscClassIdRegister("Mesh Transform",&DMPLEXTRANSFORM_CLASSID));

#if defined(PETSC_HAVE_HYPRE)
  PetscCall(MatRegister(MATHYPRESTRUCT, MatCreate_HYPREStruct));
  PetscCall(MatRegister(MATHYPRESSTRUCT, MatCreate_HYPRESStruct));
#endif
  PetscCall(PetscSectionSymRegister(PETSCSECTIONSYMLABEL,PetscSectionSymCreate_Label));

  /* Register Constructors */
  PetscCall(DMRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("DMConvert",              DM_CLASSID,&DM_Convert));
  PetscCall(PetscLogEventRegister("DMGlobalToLocal",        DM_CLASSID,&DM_GlobalToLocal));
  PetscCall(PetscLogEventRegister("DMLocalToGlobal",        DM_CLASSID,&DM_LocalToGlobal));
  PetscCall(PetscLogEventRegister("DMLocatePoints",         DM_CLASSID,&DM_LocatePoints));
  PetscCall(PetscLogEventRegister("DMCoarsen",              DM_CLASSID,&DM_Coarsen));
  PetscCall(PetscLogEventRegister("DMRefine",               DM_CLASSID,&DM_Refine));
  PetscCall(PetscLogEventRegister("DMCreateInterp",         DM_CLASSID,&DM_CreateInterpolation));
  PetscCall(PetscLogEventRegister("DMCreateRestrict",       DM_CLASSID,&DM_CreateRestriction));
  PetscCall(PetscLogEventRegister("DMCreateInject",         DM_CLASSID,&DM_CreateInjection));
  PetscCall(PetscLogEventRegister("DMCreateMat",            DM_CLASSID,&DM_CreateMatrix));
  PetscCall(PetscLogEventRegister("DMCreateMassMat",        DM_CLASSID,&DM_CreateMassMatrix));
  PetscCall(PetscLogEventRegister("DMLoad",                 DM_CLASSID,&DM_Load));
  PetscCall(PetscLogEventRegister("DMAdaptInterp",          DM_CLASSID,&DM_AdaptInterpolator));

  PetscCall(PetscLogEventRegister("DMPlexBuFrCeLi",         DM_CLASSID,&DMPLEX_BuildFromCellList));
  PetscCall(PetscLogEventRegister("DMPlexBuCoFrCeLi",       DM_CLASSID,&DMPLEX_BuildCoordinatesFromCellList));
  PetscCall(PetscLogEventRegister("DMPlexCreateGmsh",       DM_CLASSID,&DMPLEX_CreateGmsh));
  PetscCall(PetscLogEventRegister("DMPlexCrFromFile",       DM_CLASSID,&DMPLEX_CreateFromFile));
  PetscCall(PetscLogEventRegister("Mesh Partition",         DM_CLASSID,&DMPLEX_Partition));
  PetscCall(PetscLogEventRegister("Mesh Migration",         DM_CLASSID,&DMPLEX_Migrate));
  PetscCall(PetscLogEventRegister("DMPlexPartSelf",         DM_CLASSID,&DMPLEX_PartSelf));
  PetscCall(PetscLogEventRegister("DMPlexPartLblInv",       DM_CLASSID,&DMPLEX_PartLabelInvert));
  PetscCall(PetscLogEventRegister("DMPlexPartLblSF",        DM_CLASSID,&DMPLEX_PartLabelCreateSF));
  PetscCall(PetscLogEventRegister("DMPlexPartStrtSF",       DM_CLASSID,&DMPLEX_PartStratSF));
  PetscCall(PetscLogEventRegister("DMPlexPointSF",          DM_CLASSID,&DMPLEX_CreatePointSF));
  PetscCall(PetscLogEventRegister("DMPlexInterp",           DM_CLASSID,&DMPLEX_Interpolate));
  PetscCall(PetscLogEventRegister("DMPlexDistribute",       DM_CLASSID,&DMPLEX_Distribute));
  PetscCall(PetscLogEventRegister("DMPlexDistCones",        DM_CLASSID,&DMPLEX_DistributeCones));
  PetscCall(PetscLogEventRegister("DMPlexDistLabels",       DM_CLASSID,&DMPLEX_DistributeLabels));
  PetscCall(PetscLogEventRegister("DMPlexDistSF",           DM_CLASSID,&DMPLEX_DistributeSF));
  PetscCall(PetscLogEventRegister("DMPlexDistOvrlp",        DM_CLASSID,&DMPLEX_DistributeOverlap));
  PetscCall(PetscLogEventRegister("DMPlexDistField",        DM_CLASSID,&DMPLEX_DistributeField));
  PetscCall(PetscLogEventRegister("DMPlexDistData",         DM_CLASSID,&DMPLEX_DistributeData));
  PetscCall(PetscLogEventRegister("DMPlexInterpSF",         DM_CLASSID,&DMPLEX_InterpolateSF));
  PetscCall(PetscLogEventRegister("DMPlexGToNBegin",        DM_CLASSID,&DMPLEX_GlobalToNaturalBegin));
  PetscCall(PetscLogEventRegister("DMPlexGToNEnd",          DM_CLASSID,&DMPLEX_GlobalToNaturalEnd));
  PetscCall(PetscLogEventRegister("DMPlexNToGBegin",        DM_CLASSID,&DMPLEX_NaturalToGlobalBegin));
  PetscCall(PetscLogEventRegister("DMPlexNToGEnd",          DM_CLASSID,&DMPLEX_NaturalToGlobalEnd));
  PetscCall(PetscLogEventRegister("DMPlexStratify",         DM_CLASSID,&DMPLEX_Stratify));
  PetscCall(PetscLogEventRegister("DMPlexSymmetrize",       DM_CLASSID,&DMPLEX_Symmetrize));
  PetscCall(PetscLogEventRegister("DMPlexPrealloc",         DM_CLASSID,&DMPLEX_Preallocate));
  PetscCall(PetscLogEventRegister("DMPlexResidualFE",       DM_CLASSID,&DMPLEX_ResidualFEM));
  PetscCall(PetscLogEventRegister("DMPlexJacobianFE",       DM_CLASSID,&DMPLEX_JacobianFEM));
  PetscCall(PetscLogEventRegister("DMPlexInterpFE",         DM_CLASSID,&DMPLEX_InterpolatorFEM));
  PetscCall(PetscLogEventRegister("DMPlexInjectorFE",       DM_CLASSID,&DMPLEX_InjectorFEM));
  PetscCall(PetscLogEventRegister("DMPlexIntegralFEM",      DM_CLASSID,&DMPLEX_IntegralFEM));
  PetscCall(PetscLogEventRegister("DMPlexRebalance",        DM_CLASSID,&DMPLEX_RebalanceSharedPoints));
  PetscCall(PetscLogEventRegister("DMPlexLocatePoints",     DM_CLASSID,&DMPLEX_LocatePoints));
  PetscCall(PetscLogEventRegister("DMPlexTopologyView",     DM_CLASSID,&DMPLEX_TopologyView));
  PetscCall(PetscLogEventRegister("DMPlexLabelsView",       DM_CLASSID,&DMPLEX_LabelsView));
  PetscCall(PetscLogEventRegister("DMPlexCoordinatesView",  DM_CLASSID,&DMPLEX_CoordinatesView));
  PetscCall(PetscLogEventRegister("DMPlexSectionView",      DM_CLASSID,&DMPLEX_SectionView));
  PetscCall(PetscLogEventRegister("DMPlexGlobalVectorView", DM_CLASSID,&DMPLEX_GlobalVectorView));
  PetscCall(PetscLogEventRegister("DMPlexLocalVectorView",  DM_CLASSID,&DMPLEX_LocalVectorView));
  PetscCall(PetscLogEventRegister("DMPlexTopologyLoad",     DM_CLASSID,&DMPLEX_TopologyLoad));
  PetscCall(PetscLogEventRegister("DMPlexLabelsLoad",       DM_CLASSID,&DMPLEX_LabelsLoad));
  PetscCall(PetscLogEventRegister("DMPlexCoordinatesLoad",  DM_CLASSID,&DMPLEX_CoordinatesLoad));
  PetscCall(PetscLogEventRegister("DMPlexSectionLoad",      DM_CLASSID,&DMPLEX_SectionLoad));
  PetscCall(PetscLogEventRegister("DMPlexGlobalVectorLoad", DM_CLASSID,&DMPLEX_GlobalVectorLoad));
  PetscCall(PetscLogEventRegister("DMPlexLocalVectorLoad",  DM_CLASSID,&DMPLEX_LocalVectorLoad));
  PetscCall(PetscLogEventRegister("DMPlexMetricEnforceSPD", DM_CLASSID,&DMPLEX_MetricEnforceSPD));
  PetscCall(PetscLogEventRegister("DMPlexMetricNormalize",  DM_CLASSID,&DMPLEX_MetricNormalize));
  PetscCall(PetscLogEventRegister("DMPlexMetricAverage",    DM_CLASSID,&DMPLEX_MetricAverage));
  PetscCall(PetscLogEventRegister("DMPlexMetricIntersect",  DM_CLASSID,&DMPLEX_MetricIntersection));

  PetscCall(PetscLogEventRegister("RebalBuildGraph",        DM_CLASSID,&DMPLEX_RebalBuildGraph));
  PetscCall(PetscLogEventRegister("RebalGatherGraph",       DM_CLASSID,&DMPLEX_RebalGatherGraph));
  PetscCall(PetscLogEventRegister("RebalPartition",         DM_CLASSID,&DMPLEX_RebalPartition));
  PetscCall(PetscLogEventRegister("RebalScatterPart",       DM_CLASSID,&DMPLEX_RebalScatterPart));
  PetscCall(PetscLogEventRegister("RebalRewriteSF",         DM_CLASSID,&DMPLEX_RebalRewriteSF));

  PetscCall(PetscLogEventRegister("DMSwarmMigrate",         DM_CLASSID,&DMSWARM_Migrate));
  PetscCall(PetscLogEventRegister("DMSwarmDETSetup",        DM_CLASSID,&DMSWARM_DataExchangerTopologySetup));
  PetscCall(PetscLogEventRegister("DMSwarmDExBegin",        DM_CLASSID,&DMSWARM_DataExchangerBegin));
  PetscCall(PetscLogEventRegister("DMSwarmDExEnd",          DM_CLASSID,&DMSWARM_DataExchangerEnd));
  PetscCall(PetscLogEventRegister("DMSwarmDESendCnt",       DM_CLASSID,&DMSWARM_DataExchangerSendCount));
  PetscCall(PetscLogEventRegister("DMSwarmDEPack",          DM_CLASSID,&DMSWARM_DataExchangerPack));
  PetscCall(PetscLogEventRegister("DMSwarmAddPnts",         DM_CLASSID,&DMSWARM_AddPoints));
  PetscCall(PetscLogEventRegister("DMSwarmRmvPnts",         DM_CLASSID,&DMSWARM_RemovePoints));
  PetscCall(PetscLogEventRegister("DMSwarmSort",            DM_CLASSID,&DMSWARM_Sort));
  PetscCall(PetscLogEventRegister("DMSwarmSetSizes",        DM_CLASSID,&DMSWARM_SetSizes));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = DM_CLASSID;
    PetscCall(PetscInfoProcessClass("dm", 1, classids));
  }

  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("dm",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(DM_CLASSID));
  }

  PetscCall(DMGenerateRegisterAll());
  PetscCall(PetscRegisterFinalize(DMGenerateRegisterDestroy));
  PetscCall(DMPlexTransformRegisterAll());
  PetscCall(PetscRegisterFinalize(DMPlexTransformRegisterDestroy));
  PetscCall(PetscRegisterFinalize(DMFinalizePackage));
  PetscFunctionReturn(0);
}
#include <petscfe.h>

static PetscBool PetscFEPackageInitialized = PETSC_FALSE;
/*@C
  PetscFEFinalizePackage - This function finalizes everything in the PetscFE package. It is called
  from PetscFinalize().

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PetscFEFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscSpaceList));
  PetscCall(PetscFunctionListDestroy(&PetscDualSpaceList));
  PetscCall(PetscFunctionListDestroy(&PetscFEList));
  PetscFEPackageInitialized       = PETSC_FALSE;
  PetscSpaceRegisterAllCalled     = PETSC_FALSE;
  PetscDualSpaceRegisterAllCalled = PETSC_FALSE;
  PetscFERegisterAllCalled        = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscFEInitializePackage - This function initializes everything in the FE package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PetscSpaceCreate()
  when using static libraries.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PetscFEInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscFEPackageInitialized) PetscFunctionReturn(0);
  PetscFEPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  PetscCall(PetscClassIdRegister("Linear Space", &PETSCSPACE_CLASSID));
  PetscCall(PetscClassIdRegister("Dual Space",   &PETSCDUALSPACE_CLASSID));
  PetscCall(PetscClassIdRegister("FE Space",     &PETSCFE_CLASSID));
  /* Register Constructors */
  PetscCall(PetscSpaceRegisterAll());
  PetscCall(PetscDualSpaceRegisterAll());
  PetscCall(PetscFERegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("DualSpaceSetUp", PETSCDUALSPACE_CLASSID, &PETSCDUALSPACE_SetUp));
  PetscCall(PetscLogEventRegister("FESetUp",        PETSCFE_CLASSID,        &PETSCFE_SetUp));
  /* Process Info */
  {
    PetscClassId  classids[3];

    classids[0] = PETSCFE_CLASSID;
    classids[1] = PETSCSPACE_CLASSID;
    classids[2] = PETSCDUALSPACE_CLASSID;
    PetscCall(PetscInfoProcessClass("fe", 1, classids));
    PetscCall(PetscInfoProcessClass("space", 1, &classids[1]));
    PetscCall(PetscInfoProcessClass("dualspace", 1, &classids[2]));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("fe",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSCFE_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscFEFinalizePackage));
  PetscFunctionReturn(0);
}
#include <petscfv.h>

static PetscBool PetscFVPackageInitialized = PETSC_FALSE;
/*@C
  PetscFVFinalizePackage - This function finalizes everything in the PetscFV package. It is called
  from PetscFinalize().

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PetscFVFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscLimiterList));
  PetscCall(PetscFunctionListDestroy(&PetscFVList));
  PetscFVPackageInitialized     = PETSC_FALSE;
  PetscFVRegisterAllCalled      = PETSC_FALSE;
  PetscLimiterRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscFVInitializePackage - This function initializes everything in the FV package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PetscFVCreate()
  when using static libraries.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PetscFVInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscFVPackageInitialized) PetscFunctionReturn(0);
  PetscFVPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  PetscCall(PetscClassIdRegister("FV Space", &PETSCFV_CLASSID));
  PetscCall(PetscClassIdRegister("Limiter",  &PETSCLIMITER_CLASSID));
  /* Register Constructors */
  PetscCall(PetscFVRegisterAll());
  /* Register Events */
  /* Process Info */
  {
    PetscClassId  classids[2];

    classids[0] = PETSCFV_CLASSID;
    classids[1] = PETSCLIMITER_CLASSID;
    PetscCall(PetscInfoProcessClass("fv", 1, classids));
    PetscCall(PetscInfoProcessClass("limiter", 1, &classids[1]));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("fv",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSCFV_CLASSID));
    PetscCall(PetscStrInList("limiter",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSCLIMITER_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscFVFinalizePackage));
  PetscFunctionReturn(0);
}
#include <petscds.h>

static PetscBool PetscDSPackageInitialized = PETSC_FALSE;
/*@C
  PetscDSFinalizePackage - This function finalizes everything in the PetscDS package. It is called
  from PetscFinalize().

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PetscDSFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscDSList));
  PetscDSPackageInitialized = PETSC_FALSE;
  PetscDSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSInitializePackage - This function initializes everything in the DS package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PetscDSCreate()
  when using static libraries.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PetscDSInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscDSPackageInitialized) PetscFunctionReturn(0);
  PetscDSPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  PetscCall(PetscClassIdRegister("Discrete System", &PETSCDS_CLASSID));
  PetscCall(PetscClassIdRegister("Weak Form",       &PETSCWEAKFORM_CLASSID));
  /* Register Constructors */
  PetscCall(PetscDSRegisterAll());
  /* Register Events */
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSCDS_CLASSID;
    PetscCall(PetscInfoProcessClass("ds", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("ds",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSCDS_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscDSFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the mesh generators and partitioners that are in
  the basic DM library.

*/
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscdm(void)
{
  PetscFunctionBegin;
  PetscCall(AOInitializePackage());
  PetscCall(PetscPartitionerInitializePackage());
  PetscCall(DMInitializePackage());
  PetscCall(PetscFEInitializePackage());
  PetscCall(PetscFVInitializePackage());
  PetscCall(DMFieldInitializePackage());
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
