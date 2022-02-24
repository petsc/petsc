
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

.seealso: PetscInitialize()
@*/
PetscErrorCode  DMFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&DMList));
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

.seealso: PetscInitialize()
@*/
PetscErrorCode DMInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (DMPackageInitialized) PetscFunctionReturn(0);
  DMPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Distributed Mesh",&DM_CLASSID));
  CHKERRQ(PetscClassIdRegister("DM Label",&DMLABEL_CLASSID));
  CHKERRQ(PetscClassIdRegister("Quadrature",&PETSCQUADRATURE_CLASSID));
  CHKERRQ(PetscClassIdRegister("Mesh Transform",&DMPLEXTRANSFORM_CLASSID));

#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(MatRegister(MATHYPRESTRUCT, MatCreate_HYPREStruct));
  CHKERRQ(MatRegister(MATHYPRESSTRUCT, MatCreate_HYPRESStruct));
#endif
  CHKERRQ(PetscSectionSymRegister(PETSCSECTIONSYMLABEL,PetscSectionSymCreate_Label));

  /* Register Constructors */
  CHKERRQ(DMRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("DMConvert",              DM_CLASSID,&DM_Convert));
  CHKERRQ(PetscLogEventRegister("DMGlobalToLocal",        DM_CLASSID,&DM_GlobalToLocal));
  CHKERRQ(PetscLogEventRegister("DMLocalToGlobal",        DM_CLASSID,&DM_LocalToGlobal));
  CHKERRQ(PetscLogEventRegister("DMLocatePoints",         DM_CLASSID,&DM_LocatePoints));
  CHKERRQ(PetscLogEventRegister("DMCoarsen",              DM_CLASSID,&DM_Coarsen));
  CHKERRQ(PetscLogEventRegister("DMCreateInterp",         DM_CLASSID,&DM_CreateInterpolation));
  CHKERRQ(PetscLogEventRegister("DMCreateRestrict",       DM_CLASSID,&DM_CreateRestriction));
  CHKERRQ(PetscLogEventRegister("DMCreateInject",         DM_CLASSID,&DM_CreateInjection));
  CHKERRQ(PetscLogEventRegister("DMCreateMat",            DM_CLASSID,&DM_CreateMatrix));
  CHKERRQ(PetscLogEventRegister("DMLoad",                 DM_CLASSID,&DM_Load));
  CHKERRQ(PetscLogEventRegister("DMAdaptInterp",          DM_CLASSID,&DM_AdaptInterpolator));

  CHKERRQ(PetscLogEventRegister("DMPlexBuFrCeLi",         DM_CLASSID,&DMPLEX_BuildFromCellList));
  CHKERRQ(PetscLogEventRegister("DMPlexBuCoFrCeLi",       DM_CLASSID,&DMPLEX_BuildCoordinatesFromCellList));
  CHKERRQ(PetscLogEventRegister("DMPlexCreateGmsh",       DM_CLASSID,&DMPLEX_CreateGmsh));
  CHKERRQ(PetscLogEventRegister("DMPlexCrFromFile",       DM_CLASSID,&DMPLEX_CreateFromFile));
  CHKERRQ(PetscLogEventRegister("Mesh Partition",         DM_CLASSID,&DMPLEX_Partition));
  CHKERRQ(PetscLogEventRegister("Mesh Migration",         DM_CLASSID,&DMPLEX_Migrate));
  CHKERRQ(PetscLogEventRegister("DMPlexPartSelf",         DM_CLASSID,&DMPLEX_PartSelf));
  CHKERRQ(PetscLogEventRegister("DMPlexPartLblInv",       DM_CLASSID,&DMPLEX_PartLabelInvert));
  CHKERRQ(PetscLogEventRegister("DMPlexPartLblSF",        DM_CLASSID,&DMPLEX_PartLabelCreateSF));
  CHKERRQ(PetscLogEventRegister("DMPlexPartStrtSF",       DM_CLASSID,&DMPLEX_PartStratSF));
  CHKERRQ(PetscLogEventRegister("DMPlexPointSF",          DM_CLASSID,&DMPLEX_CreatePointSF));
  CHKERRQ(PetscLogEventRegister("DMPlexInterp",           DM_CLASSID,&DMPLEX_Interpolate));
  CHKERRQ(PetscLogEventRegister("DMPlexDistribute",       DM_CLASSID,&DMPLEX_Distribute));
  CHKERRQ(PetscLogEventRegister("DMPlexDistCones",        DM_CLASSID,&DMPLEX_DistributeCones));
  CHKERRQ(PetscLogEventRegister("DMPlexDistLabels",       DM_CLASSID,&DMPLEX_DistributeLabels));
  CHKERRQ(PetscLogEventRegister("DMPlexDistSF",           DM_CLASSID,&DMPLEX_DistributeSF));
  CHKERRQ(PetscLogEventRegister("DMPlexDistOvrlp",        DM_CLASSID,&DMPLEX_DistributeOverlap));
  CHKERRQ(PetscLogEventRegister("DMPlexDistField",        DM_CLASSID,&DMPLEX_DistributeField));
  CHKERRQ(PetscLogEventRegister("DMPlexDistData",         DM_CLASSID,&DMPLEX_DistributeData));
  CHKERRQ(PetscLogEventRegister("DMPlexInterpSF",         DM_CLASSID,&DMPLEX_InterpolateSF));
  CHKERRQ(PetscLogEventRegister("DMPlexGToNBegin",        DM_CLASSID,&DMPLEX_GlobalToNaturalBegin));
  CHKERRQ(PetscLogEventRegister("DMPlexGToNEnd",          DM_CLASSID,&DMPLEX_GlobalToNaturalEnd));
  CHKERRQ(PetscLogEventRegister("DMPlexNToGBegin",        DM_CLASSID,&DMPLEX_NaturalToGlobalBegin));
  CHKERRQ(PetscLogEventRegister("DMPlexNToGEnd",          DM_CLASSID,&DMPLEX_NaturalToGlobalEnd));
  CHKERRQ(PetscLogEventRegister("DMPlexStratify",         DM_CLASSID,&DMPLEX_Stratify));
  CHKERRQ(PetscLogEventRegister("DMPlexSymmetrize",       DM_CLASSID,&DMPLEX_Symmetrize));
  CHKERRQ(PetscLogEventRegister("DMPlexPrealloc",         DM_CLASSID,&DMPLEX_Preallocate));
  CHKERRQ(PetscLogEventRegister("DMPlexResidualFE",       DM_CLASSID,&DMPLEX_ResidualFEM));
  CHKERRQ(PetscLogEventRegister("DMPlexJacobianFE",       DM_CLASSID,&DMPLEX_JacobianFEM));
  CHKERRQ(PetscLogEventRegister("DMPlexInterpFE",         DM_CLASSID,&DMPLEX_InterpolatorFEM));
  CHKERRQ(PetscLogEventRegister("DMPlexInjectorFE",       DM_CLASSID,&DMPLEX_InjectorFEM));
  CHKERRQ(PetscLogEventRegister("DMPlexIntegralFEM",      DM_CLASSID,&DMPLEX_IntegralFEM));
  CHKERRQ(PetscLogEventRegister("DMPlexRebalance",        DM_CLASSID,&DMPLEX_RebalanceSharedPoints));
  CHKERRQ(PetscLogEventRegister("DMPlexLocatePoints",     DM_CLASSID,&DMPLEX_LocatePoints));
  CHKERRQ(PetscLogEventRegister("DMPlexTopologyView",     DM_CLASSID,&DMPLEX_TopologyView));
  CHKERRQ(PetscLogEventRegister("DMPlexLabelsView",       DM_CLASSID,&DMPLEX_LabelsView));
  CHKERRQ(PetscLogEventRegister("DMPlexCoordinatesView",  DM_CLASSID,&DMPLEX_CoordinatesView));
  CHKERRQ(PetscLogEventRegister("DMPlexSectionView",      DM_CLASSID,&DMPLEX_SectionView));
  CHKERRQ(PetscLogEventRegister("DMPlexGlobalVectorView", DM_CLASSID,&DMPLEX_GlobalVectorView));
  CHKERRQ(PetscLogEventRegister("DMPlexLocalVectorView",  DM_CLASSID,&DMPLEX_LocalVectorView));
  CHKERRQ(PetscLogEventRegister("DMPlexTopologyLoad",     DM_CLASSID,&DMPLEX_TopologyLoad));
  CHKERRQ(PetscLogEventRegister("DMPlexLabelsLoad",       DM_CLASSID,&DMPLEX_LabelsLoad));
  CHKERRQ(PetscLogEventRegister("DMPlexCoordinatesLoad",  DM_CLASSID,&DMPLEX_CoordinatesLoad));
  CHKERRQ(PetscLogEventRegister("DMPlexSectionLoad",      DM_CLASSID,&DMPLEX_SectionLoad));
  CHKERRQ(PetscLogEventRegister("DMPlexGlobalVectorLoad", DM_CLASSID,&DMPLEX_GlobalVectorLoad));
  CHKERRQ(PetscLogEventRegister("DMPlexLocalVectorLoad",  DM_CLASSID,&DMPLEX_LocalVectorLoad));
  CHKERRQ(PetscLogEventRegister("DMPlexMetricEnforceSPD", DM_CLASSID,&DMPLEX_MetricEnforceSPD));
  CHKERRQ(PetscLogEventRegister("DMPlexMetricNormalize",  DM_CLASSID,&DMPLEX_MetricNormalize));
  CHKERRQ(PetscLogEventRegister("DMPlexMetricAverage",    DM_CLASSID,&DMPLEX_MetricAverage));
  CHKERRQ(PetscLogEventRegister("DMPlexMetricIntersect",  DM_CLASSID,&DMPLEX_MetricIntersection));

  CHKERRQ(PetscLogEventRegister("DMSwarmMigrate",         DM_CLASSID,&DMSWARM_Migrate));
  CHKERRQ(PetscLogEventRegister("DMSwarmDETSetup",        DM_CLASSID,&DMSWARM_DataExchangerTopologySetup));
  CHKERRQ(PetscLogEventRegister("DMSwarmDExBegin",        DM_CLASSID,&DMSWARM_DataExchangerBegin));
  CHKERRQ(PetscLogEventRegister("DMSwarmDExEnd",          DM_CLASSID,&DMSWARM_DataExchangerEnd));
  CHKERRQ(PetscLogEventRegister("DMSwarmDESendCnt",       DM_CLASSID,&DMSWARM_DataExchangerSendCount));
  CHKERRQ(PetscLogEventRegister("DMSwarmDEPack",          DM_CLASSID,&DMSWARM_DataExchangerPack));
  CHKERRQ(PetscLogEventRegister("DMSwarmAddPnts",         DM_CLASSID,&DMSWARM_AddPoints));
  CHKERRQ(PetscLogEventRegister("DMSwarmRmvPnts",         DM_CLASSID,&DMSWARM_RemovePoints));
  CHKERRQ(PetscLogEventRegister("DMSwarmSort",            DM_CLASSID,&DMSWARM_Sort));
  CHKERRQ(PetscLogEventRegister("DMSwarmSetSizes",        DM_CLASSID,&DMSWARM_SetSizes));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = DM_CLASSID;
    CHKERRQ(PetscInfoProcessClass("dm", 1, classids));
  }

  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("dm",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(DM_CLASSID));
  }

  CHKERRQ(DMGenerateRegisterAll());
  CHKERRQ(PetscRegisterFinalize(DMGenerateRegisterDestroy));
  CHKERRQ(DMPlexTransformRegisterAll());
  CHKERRQ(PetscRegisterFinalize(DMPlexTransformRegisterDestroy));
  CHKERRQ(PetscRegisterFinalize(DMFinalizePackage));
  PetscFunctionReturn(0);
}
#include <petscfe.h>

static PetscBool PetscFEPackageInitialized = PETSC_FALSE;
/*@C
  PetscFEFinalizePackage - This function finalizes everything in the PetscFE package. It is called
  from PetscFinalize().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode PetscFEFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&PetscSpaceList));
  CHKERRQ(PetscFunctionListDestroy(&PetscDualSpaceList));
  CHKERRQ(PetscFunctionListDestroy(&PetscFEList));
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

.seealso: PetscInitialize()
@*/
PetscErrorCode PetscFEInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscFEPackageInitialized) PetscFunctionReturn(0);
  PetscFEPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Linear Space", &PETSCSPACE_CLASSID));
  CHKERRQ(PetscClassIdRegister("Dual Space",   &PETSCDUALSPACE_CLASSID));
  CHKERRQ(PetscClassIdRegister("FE Space",     &PETSCFE_CLASSID));
  /* Register Constructors */
  CHKERRQ(PetscSpaceRegisterAll());
  CHKERRQ(PetscDualSpaceRegisterAll());
  CHKERRQ(PetscFERegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("DualSpaceSetUp", PETSCDUALSPACE_CLASSID, &PETSCDUALSPACE_SetUp));
  CHKERRQ(PetscLogEventRegister("FESetUp",        PETSCFE_CLASSID,        &PETSCFE_SetUp));
  /* Process Info */
  {
    PetscClassId  classids[3];

    classids[0] = PETSCFE_CLASSID;
    classids[1] = PETSCSPACE_CLASSID;
    classids[2] = PETSCDUALSPACE_CLASSID;
    CHKERRQ(PetscInfoProcessClass("fe", 1, classids));
    CHKERRQ(PetscInfoProcessClass("space", 1, &classids[1]));
    CHKERRQ(PetscInfoProcessClass("dualspace", 1, &classids[2]));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("fe",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(PETSCFE_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(PetscFEFinalizePackage));
  PetscFunctionReturn(0);
}
#include <petscfv.h>

static PetscBool PetscFVPackageInitialized = PETSC_FALSE;
/*@C
  PetscFVFinalizePackage - This function finalizes everything in the PetscFV package. It is called
  from PetscFinalize().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode PetscFVFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&PetscLimiterList));
  CHKERRQ(PetscFunctionListDestroy(&PetscFVList));
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

.seealso: PetscInitialize()
@*/
PetscErrorCode PetscFVInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscFVPackageInitialized) PetscFunctionReturn(0);
  PetscFVPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("FV Space", &PETSCFV_CLASSID));
  CHKERRQ(PetscClassIdRegister("Limiter",  &PETSCLIMITER_CLASSID));
  /* Register Constructors */
  CHKERRQ(PetscFVRegisterAll());
  /* Register Events */
  /* Process Info */
  {
    PetscClassId  classids[2];

    classids[0] = PETSCFV_CLASSID;
    classids[1] = PETSCLIMITER_CLASSID;
    CHKERRQ(PetscInfoProcessClass("fv", 1, classids));
    CHKERRQ(PetscInfoProcessClass("limiter", 1, &classids[1]));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("fv",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(PETSCFV_CLASSID));
    CHKERRQ(PetscStrInList("limiter",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(PETSCLIMITER_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(PetscFVFinalizePackage));
  PetscFunctionReturn(0);
}
#include <petscds.h>

static PetscBool PetscDSPackageInitialized = PETSC_FALSE;
/*@C
  PetscDSFinalizePackage - This function finalizes everything in the PetscDS package. It is called
  from PetscFinalize().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode PetscDSFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&PetscDSList));
  PetscDSPackageInitialized = PETSC_FALSE;
  PetscDSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDSInitializePackage - This function initializes everything in the DS package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PetscDSCreate()
  when using static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode PetscDSInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscDSPackageInitialized) PetscFunctionReturn(0);
  PetscDSPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Discrete System", &PETSCDS_CLASSID));
  CHKERRQ(PetscClassIdRegister("Weak Form",       &PETSCWEAKFORM_CLASSID));
  /* Register Constructors */
  CHKERRQ(PetscDSRegisterAll());
  /* Register Events */
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSCDS_CLASSID;
    CHKERRQ(PetscInfoProcessClass("ds", 1, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("ds",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(PETSCDS_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(PetscDSFinalizePackage));
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
  CHKERRQ(AOInitializePackage());
  CHKERRQ(PetscPartitionerInitializePackage());
  CHKERRQ(DMInitializePackage());
  CHKERRQ(PetscFEInitializePackage());
  CHKERRQ(PetscFVInitializePackage());
  CHKERRQ(DMFieldInitializePackage());
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
