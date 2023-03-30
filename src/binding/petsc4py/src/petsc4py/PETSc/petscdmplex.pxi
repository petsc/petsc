# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef enum PetscDMPlexReorderDefaultFlag "DMPlexReorderDefaultFlag":
        DMPLEX_REORDER_DEFAULT_NOTSET
        DMPLEX_REORDER_DEFAULT_FALSE
        DMPLEX_REORDER_DEFAULT_TRUE

    ctypedef const char* PetscDMPlexTransformType "DMPlexTransformType"
    PetscDMPlexTransformType DMPLEXREFINEREGULAR
    PetscDMPlexTransformType DMPLEXREFINEALFELD
    PetscDMPlexTransformType DMPLEXREFINEPOWELLSABIN
    PetscDMPlexTransformType DMPLEXREFINEBOUNDARYLAYER
    PetscDMPlexTransformType DMPLEXREFINESBR
    PetscDMPlexTransformType DMPLEXREFINETOBOX
    PetscDMPlexTransformType DMPLEXREFINETOSIMPLEX
    PetscDMPlexTransformType DMPLEXREFINE1D
    PetscDMPlexTransformType DMPLEXEXTRUDE
    PetscDMPlexTransformType DMPLEXTRANSFORMFILTER

    PetscErrorCode DMPlexCreate(MPI_Comm,PetscDM*)
    PetscErrorCode DMPlexCreateCohesiveSubmesh(PetscDM,PetscBool,const char[],PetscInt,PetscDM*)
    PetscErrorCode DMPlexCreateFromCellListPetsc(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool,PetscInt[],PetscInt,PetscReal[],PetscDM*)
    #int DMPlexCreateFromDAG(PetscDM,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],const PetscInt[],const PetscScalar[])

    PetscErrorCode DMPlexGetChart(PetscDM,PetscInt*,PetscInt*)
    PetscErrorCode DMPlexSetChart(PetscDM,PetscInt,PetscInt)
    PetscErrorCode DMPlexGetConeSize(PetscDM,PetscInt,PetscInt*)
    PetscErrorCode DMPlexSetConeSize(PetscDM,PetscInt,PetscInt)
    PetscErrorCode DMPlexGetCone(PetscDM,PetscInt,const PetscInt*[])
    PetscErrorCode DMPlexSetCone(PetscDM,PetscInt,const PetscInt[])
    PetscErrorCode DMPlexInsertCone(PetscDM,PetscInt,PetscInt,PetscInt)
    PetscErrorCode DMPlexInsertConeOrientation(PetscDM,PetscInt,PetscInt,PetscInt)
    PetscErrorCode DMPlexGetConeOrientation(PetscDM,PetscInt,const PetscInt*[])
    PetscErrorCode DMPlexSetConeOrientation(PetscDM,PetscInt,const PetscInt[])
    PetscErrorCode DMPlexSetCellType(PetscDM,PetscInt,PetscDMPolytopeType)
    PetscErrorCode DMPlexGetCellType(PetscDM,PetscInt,PetscDMPolytopeType*)
    PetscErrorCode DMPlexGetCellTypeLabel(PetscDM,PetscDMLabel*)
    PetscErrorCode DMPlexGetSupportSize(PetscDM,PetscInt,PetscInt*)
    PetscErrorCode DMPlexSetSupportSize(PetscDM,PetscInt,PetscInt)
    PetscErrorCode DMPlexGetSupport(PetscDM,PetscInt,const PetscInt*[])
    PetscErrorCode DMPlexSetSupport(PetscDM,PetscInt,const PetscInt[])
    #int DMPlexInsertSupport(PetscDM,PetscInt,PetscInt,PetscInt)
    #int DMPlexGetConeSection(PetscDM,PetscSection*)
    #int DMPlexGetSupportSection(PetscDM,PetscSection*)
    #int DMPlexGetCones(PetscDM,PetscInt*[])
    #int DMPlexGetConeOrientations(PetscDM,PetscInt*[])
    PetscErrorCode DMPlexGetMaxSizes(PetscDM,PetscInt*,PetscInt*)
    PetscErrorCode DMPlexSymmetrize(PetscDM)
    PetscErrorCode DMPlexStratify(PetscDM)
    #int DMPlexEqual(PetscDM,PetscDM,PetscBool*)
    PetscErrorCode DMPlexOrient(PetscDM)
    PetscErrorCode DMPlexInterpolate(PetscDM,PetscDM*)
    PetscErrorCode DMPlexUninterpolate(PetscDM,PetscDM*)
    #int DMPlexLoad(PetscViewer,PetscDM)
    #int DMPlexSetPreallocationCenterDimension(PetscDM,PetscInt)
    #int DMPlexGetPreallocationCenterDimension(PetscDM,PetscInt*)
    #int DMPlexPreallocateOperator(PetscDM,PetscInt,PetscSection,PetscSection,PetscInt[],PetscInt[],PetscInt[],PetscInt[],Mat,PetscBool)
    PetscErrorCode DMPlexGetPointLocal(PetscDM,PetscInt,PetscInt*,PetscInt*)
    #int DMPlexPointLocalRef(PetscDM,PetscInt,PetscScalar*,void*)
    #int DMPlexPointLocalRead(PetscDM,PetscInt,const PetscScalar*,const void*)
    PetscErrorCode DMPlexGetPointGlobal(PetscDM,PetscInt,PetscInt*,PetscInt*)
    #int DMPlexPointGlobalRef(PetscDM,PetscInt,PetscScalar*,void*)
    #int DMPlexPointGlobalRead(PetscDM,PetscInt,const PetscScalar*,const void*)
    PetscErrorCode DMPlexGetPointLocalField(PetscDM,PetscInt,PetscInt,PetscInt*,PetscInt*)
    PetscErrorCode DMPlexGetPointGlobalField(PetscDM,PetscInt,PetscInt,PetscInt*,PetscInt*)
    PetscErrorCode DMPlexCreateClosureIndex(PetscDM,PetscSection)
    #int PetscSectionCreateGlobalSectionLabel(PetscSection,PetscSF,PetscBool,PetscDMLabel,PetscInt,PetscSection*)

    PetscErrorCode DMPlexGetCellNumbering(PetscDM,PetscIS*)
    PetscErrorCode DMPlexGetVertexNumbering(PetscDM,PetscIS*)
    PetscErrorCode DMPlexCreatePointNumbering(PetscDM,PetscIS*)

    PetscErrorCode DMPlexGetDepth(PetscDM,PetscInt*)
    #int DMPlexGetDepthLabel(PetscDM,PetscDMLabel*)
    PetscErrorCode DMPlexGetDepthStratum(PetscDM,PetscInt,PetscInt*,PetscInt*)
    PetscErrorCode DMPlexGetHeightStratum(PetscDM,PetscInt,PetscInt*,PetscInt*)
    PetscErrorCode DMPlexGetPointDepth(PetscDM,PetscInt,PetscInt*)
    PetscErrorCode DMPlexGetPointHeight(PetscDM,PetscInt,PetscInt*)

    PetscErrorCode DMPlexGetMeet(PetscDM,PetscInt,const PetscInt[],PetscInt*,const PetscInt**)
    #int DMPlexGetFullMeet(PetscDM,PetscInt,const PetscInt[],PetscInt*,const PetscInt**)
    PetscErrorCode DMPlexRestoreMeet(PetscDM,PetscInt,const PetscInt[],PetscInt*,const PetscInt**)
    PetscErrorCode DMPlexGetJoin(PetscDM,PetscInt,const PetscInt[],PetscInt*,const PetscInt**)
    PetscErrorCode DMPlexGetFullJoin(PetscDM,PetscInt,const PetscInt[],PetscInt*,const PetscInt**)
    PetscErrorCode DMPlexRestoreJoin(PetscDM,PetscInt,const PetscInt[],PetscInt*,const PetscInt**)
    PetscErrorCode DMPlexGetTransitiveClosure(PetscDM,PetscInt,PetscBool,PetscInt*,PetscInt*[])
    PetscErrorCode DMPlexRestoreTransitiveClosure(PetscDM,PetscInt,PetscBool,PetscInt*,PetscInt*[])
    PetscErrorCode DMPlexVecGetClosure(PetscDM,PetscSection,PetscVec,PetscInt,PetscInt*,PetscScalar*[])
    PetscErrorCode DMPlexVecRestoreClosure(PetscDM,PetscSection,PetscVec,PetscInt,PetscInt*,PetscScalar*[])
    PetscErrorCode DMPlexVecSetClosure(PetscDM,PetscSection,PetscVec,PetscInt,PetscScalar[],PetscInsertMode)
    PetscErrorCode DMPlexMatSetClosure(PetscDM,PetscSection,PetscSection,PetscMat,PetscInt,PetscScalar[],PetscInsertMode)

    PetscErrorCode DMPlexGenerate(PetscDM,const char[],PetscBool ,PetscDM*)
    PetscErrorCode DMPlexTriangleSetOptions(PetscDM,const char*)
    PetscErrorCode DMPlexTetgenSetOptions(PetscDM,const char*)
    #int DMPlexCopyCoordinates(PetscDM,PetscDM)
    #int DMPlexCreateDoublet(MPI_Comm,PetscInt,PetscBool,PetscBool,PetscBool,PetscReal,PetscDM*)
    PetscErrorCode DMPlexCreateBoxMesh(MPI_Comm,PetscInt,PetscBool,PetscInt[],PetscReal[],PetscReal[],PetscDMBoundaryType[],PetscBool,PetscDM*)
    PetscErrorCode DMPlexCreateBoxSurfaceMesh(MPI_Comm,PetscInt,PetscInt[],PetscReal[],PetscReal[],PetscBool,PetscDM*)
    PetscErrorCode DMPlexCreateFromFile(MPI_Comm,const char[],const char[],PetscBool,PetscDM*)
    PetscErrorCode DMPlexCreateCGNS(MPI_Comm,PetscInt,PetscBool,PetscDM*)
    PetscErrorCode DMPlexCreateCGNSFromFile(MPI_Comm,const char[],PetscBool,PetscDM*)
    PetscErrorCode DMPlexCreateExodus(MPI_Comm,PetscInt,PetscBool,PetscDM*)
    PetscErrorCode DMPlexCreateExodusFromFile(MPI_Comm,const char[],PetscBool,PetscDM*)
    PetscErrorCode DMPlexCreateGmsh(MPI_Comm,PetscViewer,PetscBool,PetscDM*)

    #int DMPlexCreateConeSection(PetscDM,PetscSection*)
    #int DMPlexInvertCell(PetscInt,PetscInt,int[])
    #int DMPlexCheckSymmetry(PetscDM)
    #int DMPlexCheckSkeleton(PetscDM,PetscBool,PetscInt)
    #int DMPlexCheckFaces(PetscDM,PetscBool,PetscInt)

    PetscErrorCode DMPlexSetAdjacencyUseAnchors(PetscDM,PetscBool)
    PetscErrorCode DMPlexGetAdjacencyUseAnchors(PetscDM,PetscBool*)
    PetscErrorCode DMPlexGetAdjacency(PetscDM,PetscInt,PetscInt*,PetscInt*[])
    #int DMPlexCreateNeighborCSR(PetscDM,PetscInt,PetscInt*,PetscInt**,PetscInt**)
    PetscErrorCode DMPlexRebalanceSharedPoints(PetscDM,PetscInt,PetscBool,PetscBool,PetscBool*)
    PetscErrorCode DMPlexDistribute(PetscDM,PetscInt,PetscSF*,PetscDM*)
    PetscErrorCode DMPlexDistributeOverlap(PetscDM,PetscInt,PetscSF*,PetscDM*)
    PetscErrorCode DMPlexDistributeGetDefault(PetscDM,PetscBool*)
    PetscErrorCode DMPlexDistributeSetDefault(PetscDM,PetscBool)
    PetscErrorCode DMPlexSetPartitioner(PetscDM,PetscPartitioner)
    PetscErrorCode DMPlexGetPartitioner(PetscDM,PetscPartitioner*)
    PetscErrorCode DMPlexDistributeField(PetscDM,PetscSF,PetscSection,PetscVec,PetscSection,PetscVec)
    #int DMPlexDistributeData(PetscDM,PetscSF,PetscSection,MPI_Datatype,void*,PetscSection,void**)
    PetscErrorCode DMPlexIsDistributed(PetscDM,PetscBool*)
    PetscErrorCode DMPlexIsSimplex(PetscDM,PetscBool*)
    PetscErrorCode DMPlexDistributionSetName(PetscDM,const char[])
    PetscErrorCode DMPlexDistributionGetName(PetscDM,const char*[])

    PetscErrorCode DMPlexGetOrdering(PetscDM,PetscMatOrderingType,PetscDMLabel,PetscIS*)
    PetscErrorCode DMPlexPermute(PetscDM,PetscIS,PetscDM*)
    PetscErrorCode DMPlexReorderGetDefault(PetscDM,PetscDMPlexReorderDefaultFlag*)
    PetscErrorCode DMPlexReorderSetDefault(PetscDM,PetscDMPlexReorderDefaultFlag)

    #int DMPlexCreateSubmesh(PetscDM,PetscDMLabel,PetscInt,PetscDM*)
    #int DMPlexCreateHybridMesh(PetscDM,PetscDMLabel,PetscDMLabel,PetscInt,PetscDMLabel*,PetscDMLabel*,PetscDM *,PetscDM *)
    #int DMPlexGetSubpointMap(PetscDM,PetscDMLabel*)
    #int DMPlexSetSubpointMap(PetscDM,PetscDMLabel)
    #int DMPlexCreateSubpointIS(PetscDM,PetscIS*)

    PetscErrorCode DMPlexCreateCoarsePointIS(PetscDM,PetscIS*)
    PetscErrorCode DMPlexMarkBoundaryFaces(PetscDM,PetscInt,PetscDMLabel)
    PetscErrorCode DMPlexLabelComplete(PetscDM,PetscDMLabel)
    PetscErrorCode DMPlexLabelCohesiveComplete(PetscDM,PetscDMLabel,PetscDMLabel,PetscInt,PetscBool,PetscDM)

    PetscErrorCode DMPlexGetRefinementLimit(PetscDM,PetscReal*)
    PetscErrorCode DMPlexSetRefinementLimit(PetscDM,PetscReal)
    PetscErrorCode DMPlexGetRefinementUniform(PetscDM,PetscBool*)
    PetscErrorCode DMPlexSetRefinementUniform(PetscDM,PetscBool)

    PetscErrorCode DMPlexGetMinRadius(PetscDM, PetscReal*)
    #int DMPlexGetNumFaceVertices(PetscDM,PetscInt,PetscInt,PetscInt*)
    #int DMPlexGetOrientedFace(PetscDM,PetscInt,PetscInt,const PetscInt[],PetscInt,PetscInt[],PetscInt[],PetscInt[],PetscBool*)

    PetscErrorCode DMPlexCreateSection(PetscDM,PetscDMLabel[],const PetscInt[],const PetscInt[],PetscInt,const PetscInt[],const PetscIS[],const PetscIS[],PetscIS,PetscSection*)

    PetscErrorCode DMPlexComputeCellGeometryFVM(PetscDM,PetscInt,PetscReal*,PetscReal[],PetscReal[])
    PetscErrorCode DMPlexConstructGhostCells(PetscDM,const char[],PetscInt*,PetscDM*)

    PetscErrorCode DMPlexMetricSetFromOptions(PetscDM)
    PetscErrorCode DMPlexMetricSetUniform(PetscDM,PetscBool)
    PetscErrorCode DMPlexMetricIsUniform(PetscDM,PetscBool*)
    PetscErrorCode DMPlexMetricSetIsotropic(PetscDM,PetscBool)
    PetscErrorCode DMPlexMetricIsIsotropic(PetscDM,PetscBool*)
    PetscErrorCode DMPlexMetricSetRestrictAnisotropyFirst(PetscDM,PetscBool)
    PetscErrorCode DMPlexMetricRestrictAnisotropyFirst(PetscDM,PetscBool*)
    PetscErrorCode DMPlexMetricSetNoInsertion(PetscDM,PetscBool)
    PetscErrorCode DMPlexMetricNoInsertion(PetscDM,PetscBool*)
    PetscErrorCode DMPlexMetricSetNoSwapping(PetscDM,PetscBool)
    PetscErrorCode DMPlexMetricNoSwapping(PetscDM,PetscBool*)
    PetscErrorCode DMPlexMetricSetNoMovement(PetscDM,PetscBool)
    PetscErrorCode DMPlexMetricNoMovement(PetscDM,PetscBool*)
    PetscErrorCode DMPlexMetricSetNoSurf(PetscDM,PetscBool)
    PetscErrorCode DMPlexMetricNoSurf(PetscDM,PetscBool*)
    PetscErrorCode DMPlexMetricSetVerbosity(PetscDM,PetscInt)
    PetscErrorCode DMPlexMetricGetVerbosity(PetscDM,PetscInt*)
    PetscErrorCode DMPlexMetricSetNumIterations(PetscDM,PetscInt)
    PetscErrorCode DMPlexMetricGetNumIterations(PetscDM,PetscInt*)
    PetscErrorCode DMPlexMetricSetMinimumMagnitude(PetscDM,PetscReal)
    PetscErrorCode DMPlexMetricGetMinimumMagnitude(PetscDM,PetscReal*)
    PetscErrorCode DMPlexMetricSetMaximumMagnitude(PetscDM,PetscReal)
    PetscErrorCode DMPlexMetricGetMaximumMagnitude(PetscDM,PetscReal*)
    PetscErrorCode DMPlexMetricSetMaximumAnisotropy(PetscDM,PetscReal)
    PetscErrorCode DMPlexMetricGetMaximumAnisotropy(PetscDM,PetscReal*)
    PetscErrorCode DMPlexMetricSetTargetComplexity(PetscDM,PetscReal)
    PetscErrorCode DMPlexMetricGetTargetComplexity(PetscDM,PetscReal*)
    PetscErrorCode DMPlexMetricSetNormalizationOrder(PetscDM,PetscReal)
    PetscErrorCode DMPlexMetricGetNormalizationOrder(PetscDM,PetscReal*)
    PetscErrorCode DMPlexMetricSetGradationFactor(PetscDM,PetscReal)
    PetscErrorCode DMPlexMetricGetGradationFactor(PetscDM,PetscReal*)
    PetscErrorCode DMPlexMetricSetHausdorffNumber(PetscDM,PetscReal)
    PetscErrorCode DMPlexMetricGetHausdorffNumber(PetscDM,PetscReal*)
    PetscErrorCode DMPlexMetricCreate(PetscDM,PetscInt,PetscVec*)
    PetscErrorCode DMPlexMetricCreateUniform(PetscDM,PetscInt,PetscReal,PetscVec*)
    PetscErrorCode DMPlexMetricCreateIsotropic(PetscDM,PetscInt,PetscVec,PetscVec*)
    PetscErrorCode DMPlexMetricDeterminantCreate(PetscDM,PetscInt,PetscVec*,PetscDM*)
    PetscErrorCode DMPlexMetricEnforceSPD(PetscDM,PetscVec,PetscBool,PetscBool,PetscVec,PetscVec)
    PetscErrorCode DMPlexMetricNormalize(PetscDM,PetscVec,PetscBool,PetscBool,PetscVec,PetscVec)
    PetscErrorCode DMPlexMetricAverage2(PetscDM,PetscVec,PetscVec,PetscVec)
    PetscErrorCode DMPlexMetricAverage3(PetscDM,PetscVec,PetscVec,PetscVec,PetscVec)
    PetscErrorCode DMPlexMetricIntersection2(PetscDM,PetscVec,PetscVec,PetscVec)
    PetscErrorCode DMPlexMetricIntersection3(PetscDM,PetscVec,PetscVec,PetscVec,PetscVec)

    PetscErrorCode DMPlexComputeGradientClementInterpolant(PetscDM,PetscVec,PetscVec)

    PetscErrorCode DMPlexTopologyView(PetscDM,PetscViewer)
    PetscErrorCode DMPlexCoordinatesView(PetscDM,PetscViewer)
    PetscErrorCode DMPlexLabelsView(PetscDM,PetscViewer)
    PetscErrorCode DMPlexSectionView(PetscDM,PetscViewer,PetscDM)
    PetscErrorCode DMPlexGlobalVectorView(PetscDM,PetscViewer,PetscDM,PetscVec)
    PetscErrorCode DMPlexLocalVectorView(PetscDM,PetscViewer,PetscDM,PetscVec)

    PetscErrorCode DMPlexTopologyLoad(PetscDM,PetscViewer,PetscSF*)
    PetscErrorCode DMPlexCoordinatesLoad(PetscDM,PetscViewer,PetscSF)
    PetscErrorCode DMPlexLabelsLoad(PetscDM,PetscViewer,PetscSF)
    PetscErrorCode DMPlexSectionLoad(PetscDM,PetscViewer,PetscDM,PetscSF,PetscSF*,PetscSF*)
    PetscErrorCode DMPlexGlobalVectorLoad(PetscDM,PetscViewer,PetscDM,PetscSF,PetscVec)
    PetscErrorCode DMPlexLocalVectorLoad(PetscDM,PetscViewer,PetscDM,PetscSF,PetscVec)

    PetscErrorCode DMPlexTransformApply(PetscDMPlexTransform, PetscDM, PetscDM *);
    PetscErrorCode DMPlexTransformCreate(MPI_Comm, PetscDMPlexTransform *);
    PetscErrorCode DMPlexTransformDestroy(PetscDMPlexTransform*);
    PetscErrorCode DMPlexTransformGetType(PetscDMPlexTransform, PetscDMPlexTransformType *);
    PetscErrorCode DMPlexTransformSetType(PetscDMPlexTransform tr, PetscDMPlexTransformType method);
    PetscErrorCode DMPlexTransformSetFromOptions(PetscDMPlexTransform);
    PetscErrorCode DMPlexTransformSetDM(PetscDMPlexTransform, PetscDM);
    PetscErrorCode DMPlexTransformSetUp(PetscDMPlexTransform);
    PetscErrorCode DMPlexTransformView(PetscDMPlexTransform tr, PetscViewer v);
    