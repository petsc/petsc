
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&DMList);CHKERRQ(ierr);
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
PetscErrorCode  DMInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (DMPackageInitialized) PetscFunctionReturn(0);
  DMPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  ierr = PetscClassIdRegister("Distributed Mesh",&DM_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("DM Label",&DMLABEL_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Quadrature",&PETSCQUADRATURE_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Mesh Transform",&DMPLEXTRANSFORM_CLASSID);CHKERRQ(ierr);

#if defined(PETSC_HAVE_HYPRE)
  ierr = MatRegister(MATHYPRESTRUCT, MatCreate_HYPREStruct);CHKERRQ(ierr);
  ierr = MatRegister(MATHYPRESSTRUCT, MatCreate_HYPRESStruct);CHKERRQ(ierr);
#endif
  ierr = PetscSectionSymRegister(PETSCSECTIONSYMLABEL,PetscSectionSymCreate_Label);CHKERRQ(ierr);

  /* Register Constructors */
  ierr = DMRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("DMConvert",              DM_CLASSID,&DM_Convert);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMGlobalToLocal",        DM_CLASSID,&DM_GlobalToLocal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMLocalToGlobal",        DM_CLASSID,&DM_LocalToGlobal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMLocatePoints",         DM_CLASSID,&DM_LocatePoints);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMCoarsen",              DM_CLASSID,&DM_Coarsen);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMCreateInterp",         DM_CLASSID,&DM_CreateInterpolation);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMCreateRestrict",       DM_CLASSID,&DM_CreateRestriction);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMCreateInject",         DM_CLASSID,&DM_CreateInjection);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMCreateMat",            DM_CLASSID,&DM_CreateMatrix);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMLoad",                 DM_CLASSID,&DM_Load);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMAdaptInterp",          DM_CLASSID,&DM_AdaptInterpolator);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("DMPlexBuFrCeLi",         DM_CLASSID,&DMPLEX_BuildFromCellList);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexBuCoFrCeLi",       DM_CLASSID,&DMPLEX_BuildCoordinatesFromCellList);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexCreateGmsh",       DM_CLASSID,&DMPLEX_CreateGmsh);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexCrFromFile",       DM_CLASSID,&DMPLEX_CreateFromFile);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Mesh Partition",         DM_CLASSID,&DMPLEX_Partition);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Mesh Migration",         DM_CLASSID,&DMPLEX_Migrate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexPartSelf",         DM_CLASSID,&DMPLEX_PartSelf);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexPartLblInv",       DM_CLASSID,&DMPLEX_PartLabelInvert);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexPartLblSF",        DM_CLASSID,&DMPLEX_PartLabelCreateSF);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexPartStrtSF",       DM_CLASSID,&DMPLEX_PartStratSF);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexPointSF",          DM_CLASSID,&DMPLEX_CreatePointSF);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexInterp",           DM_CLASSID,&DMPLEX_Interpolate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistribute",       DM_CLASSID,&DMPLEX_Distribute);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistCones",        DM_CLASSID,&DMPLEX_DistributeCones);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistLabels",       DM_CLASSID,&DMPLEX_DistributeLabels);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistSF",           DM_CLASSID,&DMPLEX_DistributeSF);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistOvrlp",        DM_CLASSID,&DMPLEX_DistributeOverlap);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistField",        DM_CLASSID,&DMPLEX_DistributeField);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistData",         DM_CLASSID,&DMPLEX_DistributeData);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexInterpSF",         DM_CLASSID,&DMPLEX_InterpolateSF);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexGToNBegin",        DM_CLASSID,&DMPLEX_GlobalToNaturalBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexGToNEnd",          DM_CLASSID,&DMPLEX_GlobalToNaturalEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexNToGBegin",        DM_CLASSID,&DMPLEX_NaturalToGlobalBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexNToGEnd",          DM_CLASSID,&DMPLEX_NaturalToGlobalEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexStratify",         DM_CLASSID,&DMPLEX_Stratify);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexSymmetrize",       DM_CLASSID,&DMPLEX_Symmetrize);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexPrealloc",         DM_CLASSID,&DMPLEX_Preallocate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexResidualFE",       DM_CLASSID,&DMPLEX_ResidualFEM);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexJacobianFE",       DM_CLASSID,&DMPLEX_JacobianFEM);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexInterpFE",         DM_CLASSID,&DMPLEX_InterpolatorFEM);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexInjectorFE",       DM_CLASSID,&DMPLEX_InjectorFEM);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexIntegralFEM",      DM_CLASSID,&DMPLEX_IntegralFEM);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexRebalance",        DM_CLASSID,&DMPLEX_RebalanceSharedPoints);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexLocatePoints",     DM_CLASSID,&DMPLEX_LocatePoints);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexTopologyView",     DM_CLASSID,&DMPLEX_TopologyView);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexLabelsView",       DM_CLASSID,&DMPLEX_LabelsView);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexCoordinatesView",  DM_CLASSID,&DMPLEX_CoordinatesView);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexSectionView",      DM_CLASSID,&DMPLEX_SectionView);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexGlobalVectorView", DM_CLASSID,&DMPLEX_GlobalVectorView);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexLocalVectorView",  DM_CLASSID,&DMPLEX_LocalVectorView);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexTopologyLoad",     DM_CLASSID,&DMPLEX_TopologyLoad);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexLabelsLoad",       DM_CLASSID,&DMPLEX_LabelsLoad);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexCoordinatesLoad",  DM_CLASSID,&DMPLEX_CoordinatesLoad);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexSectionLoad",      DM_CLASSID,&DMPLEX_SectionLoad);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexGlobalVectorLoad", DM_CLASSID,&DMPLEX_GlobalVectorLoad);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexLocalVectorLoad",  DM_CLASSID,&DMPLEX_LocalVectorLoad);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("DMSwarmMigrate",         DM_CLASSID,&DMSWARM_Migrate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMSwarmDETSetup",        DM_CLASSID,&DMSWARM_DataExchangerTopologySetup);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMSwarmDExBegin",        DM_CLASSID,&DMSWARM_DataExchangerBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMSwarmDExEnd",          DM_CLASSID,&DMSWARM_DataExchangerEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMSwarmDESendCnt",       DM_CLASSID,&DMSWARM_DataExchangerSendCount);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMSwarmDEPack",          DM_CLASSID,&DMSWARM_DataExchangerPack);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMSwarmAddPnts",         DM_CLASSID,&DMSWARM_AddPoints);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMSwarmRmvPnts",         DM_CLASSID,&DMSWARM_RemovePoints);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMSwarmSort",            DM_CLASSID,&DMSWARM_Sort);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMSwarmSetSizes",        DM_CLASSID,&DMSWARM_SetSizes);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = DM_CLASSID;
    ierr = PetscInfoProcessClass("dm", 1, classids);CHKERRQ(ierr);
  }

  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("dm",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(DM_CLASSID);CHKERRQ(ierr);}
  }

  ierr = DMPlexGenerateRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(DMPlexGenerateRegisterDestroy);CHKERRQ(ierr);
  ierr = DMPlexTransformRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(DMPlexTransformRegisterDestroy);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(DMFinalizePackage);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscSpaceList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscDualSpaceList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscFEList);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscFEPackageInitialized) PetscFunctionReturn(0);
  PetscFEPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  ierr = PetscClassIdRegister("Linear Space", &PETSCSPACE_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Dual Space",   &PETSCDUALSPACE_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("FE Space",     &PETSCFE_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);
  ierr = PetscDualSpaceRegisterAll();CHKERRQ(ierr);
  ierr = PetscFERegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("DualSpaceSetUp", PETSCDUALSPACE_CLASSID, &PETSCDUALSPACE_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("FESetUp",        PETSCFE_CLASSID,        &PETSCFE_SetUp);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[3];

    classids[0] = PETSCFE_CLASSID;
    classids[1] = PETSCSPACE_CLASSID;
    classids[2] = PETSCDUALSPACE_CLASSID;
    ierr = PetscInfoProcessClass("fe", 1, classids);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("space", 1, &classids[1]);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("dualspace", 1, &classids[2]);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("fe",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSCFE_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscFEFinalizePackage);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscLimiterList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscFVList);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscFVPackageInitialized) PetscFunctionReturn(0);
  PetscFVPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  ierr = PetscClassIdRegister("FV Space", &PETSCFV_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Limiter",  &PETSCLIMITER_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscFVRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  /* Process Info */
  {
    PetscClassId  classids[2];

    classids[0] = PETSCFV_CLASSID;
    classids[1] = PETSCLIMITER_CLASSID;
    ierr = PetscInfoProcessClass("fv", 1, classids);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("limiter", 1, &classids[1]);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("fv",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSCFV_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("limiter",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSCLIMITER_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscFVFinalizePackage);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscDSList);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscDSPackageInitialized) PetscFunctionReturn(0);
  PetscDSPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  ierr = PetscClassIdRegister("Discrete System", &PETSCDS_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Weak Form",       &PETSCWEAKFORM_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscDSRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSCDS_CLASSID;
    ierr = PetscInfoProcessClass("ds", 1, classids);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("ds",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSCDS_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscDSFinalizePackage);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AOInitializePackage();CHKERRQ(ierr);
  ierr = PetscPartitionerInitializePackage();CHKERRQ(ierr);
  ierr = DMInitializePackage();CHKERRQ(ierr);
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);
  ierr = PetscFVInitializePackage();CHKERRQ(ierr);
  ierr = DMFieldInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
