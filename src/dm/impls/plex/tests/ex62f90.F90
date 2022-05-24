program ex62f90
#include "petsc/finclude/petsc.h"
    use petsc
    implicit none
#include "exodusII.inc"

    ! Get the fortran kind associated with PetscInt and PetscReal so that we can use literal constants.
    PetscInt                           :: dummyPetscInt
    PetscReal                          :: dummyPetscreal
    integer,parameter                  :: kPI = kind(dummyPetscInt)
    integer,parameter                  :: kPR = kind(dummyPetscReal)

    type(tDM)                          :: dm,dmU,dmA,dmS,dmUA,dmUA2,pDM
    type(tDM),dimension(:),pointer     :: dmList
    type(tVec)                         :: X,U,A,S,UA,UA2
    type(tIS)                          :: isU,isA,isS,isUA
    type(tPetscSection)                :: section
    PetscInt,dimension(1)              :: fieldU = [0]
    PetscInt,dimension(1)              :: fieldA = [2]
    PetscInt,dimension(1)              :: fieldS = [1]
    PetscInt,dimension(2)              :: fieldUA = [0,2]
    character(len=PETSC_MAX_PATH_LEN)  :: ifilename,ofilename,IOBuffer
    integer                            :: exoid = -1
    type(tIS)                          :: csIS
    PetscInt,dimension(:),pointer      :: csID
    PetscInt,dimension(:),pointer      :: pStartDepth,pEndDepth
    PetscInt                           :: order = 1
    PetscInt                           :: sdim,d,pStart,pEnd,p,numCS,set,i,j
    PetscMPIInt                        :: rank,numProc
    PetscBool                          :: flg
    PetscErrorCode                     :: ierr
    MPI_Comm                           :: comm
    type(tPetscViewer)                 :: viewer

    Character(len=MXSTLN)              :: sJunk
    PetscInt                           :: numstep = 3, step
    PetscInt                           :: numNodalVar,numZonalVar
    character(len=MXSTLN)              :: nodalVarName(4)
    character(len=MXSTLN)              :: zonalVarName(6)
    logical,dimension(:,:),pointer     :: truthtable

    type(tIS)                          :: cellIS
    PetscInt,dimension(:),pointer      :: cellID
    PetscInt                           :: numCells, cell, closureSize
    PetscInt,dimension(:),pointer      :: closureA,closure

    type(tPetscSection)                :: sectionUA,coordSection
    type(tVec)                         :: UALoc,coord
    PetscScalar,dimension(:),pointer   :: cval,xyz
    PetscInt                           :: dofUA,offUA,c

    ! dof layout ordered by increasing height in the DAG: cell, face, edge, vertex
    PetscInt,dimension(3),target        :: dofS2D     = [0, 0, 3]
    PetscInt,dimension(3),target        :: dofUP1Tri  = [2, 0, 0]
    PetscInt,dimension(3),target        :: dofAP1Tri  = [1, 0, 0]
    PetscInt,dimension(3),target        :: dofUP2Tri  = [2, 2, 0]
    PetscInt,dimension(3),target        :: dofAP2Tri  = [1, 1, 0]
    PetscInt,dimension(3),target        :: dofUP1Quad = [2, 0, 0]
    PetscInt,dimension(3),target        :: dofAP1Quad = [1, 0, 0]
    PetscInt,dimension(3),target        :: dofUP2Quad = [2, 2, 2]
    PetscInt,dimension(3),target        :: dofAP2Quad = [1, 1, 1]
    PetscInt,dimension(4),target        :: dofS3D     = [0, 0, 0, 6]
    PetscInt,dimension(4),target        :: dofUP1Tet  = [3, 0, 0, 0]
    PetscInt,dimension(4),target        :: dofAP1Tet  = [1, 0, 0, 0]
    PetscInt,dimension(4),target        :: dofUP2Tet  = [3, 3, 0, 0]
    PetscInt,dimension(4),target        :: dofAP2Tet  = [1, 1, 0, 0]
    PetscInt,dimension(4),target        :: dofUP1Hex  = [3, 0, 0, 0]
    PetscInt,dimension(4),target        :: dofAP1Hex  = [1, 0, 0, 0]
    PetscInt,dimension(4),target        :: dofUP2Hex  = [3, 3, 3, 3]
    PetscInt,dimension(4),target        :: dofAP2Hex  = [1, 1, 1, 1]
    PetscInt,dimension(:),pointer       :: dofU,dofA,dofS

    type(tPetscSF)                      :: migrationSF
    PetscPartitioner                    :: part

    type(tVec)                          :: tmpVec
    PetscReal                           :: norm
    PetscReal                           :: time = 1.234_kPR

    PetscCallA(PetscInitialize(ierr))

    PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
    PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,numProc,ierr))
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-i",ifilename,flg,ierr))
    if (.not. flg) then
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"missing input file name -i <input file name>")
    end if
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-o",ofilename,flg,ierr))
    if (.not. flg) then
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"missing output file name -o <output file name>")
    end if
    PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-order",order,flg,ierr))
    if ((order > 2) .or. (order < 1)) then
        write(IOBuffer,'("Unsupported polynomial order ", I2, " not in [1,2]")') order
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,IOBuffer)
    end if

    ! Read the mesh in any supported format
    PetscCallA(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename,PETSC_NULL_CHARACTER,PETSC_TRUE,dm,ierr))
    PetscCallA(DMPlexDistributeSetDefault(dm,PETSC_FALSE,ierr))
    PetscCallA(DMSetFromOptions(dm,ierr))
    PetscCallA(DMGetDimension(dm, sdim,ierr))
    PetscCallA(DMViewFromOptions(dm, PETSC_NULL_OPTIONS,"-dm_view",ierr))

    ! Create the exodus result file

    ! enable exodus debugging information
    PetscCallA(exopts(EXVRBS+EXDEBG,ierr))
    ! Create the exodus file
    PetscCallA(PetscViewerExodusIIOpen(PETSC_COMM_WORLD,ofilename,FILE_MODE_WRITE,viewer,ierr))
    ! The long way would be
    !
    ! PetscCallA(PetscViewerCreate(PETSC_COMM_WORLD,viewer,ierr))
    ! PetscCallA(PetscViewerSetType(viewer,PETSCVIEWEREXODUSII,ierr))
    ! PetscCallA(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE,ierr))
    ! PetscCallA(PetscViewerFileSetName(viewer,ofilename,ierr))

    ! set the mesh order
    PetscCallA(PetscViewerExodusIISetOrder(viewer,order,ierr))
    PetscCallA(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD,ierr))
    !
    !    Notice how the exodus file is actually NOT open at this point (exoid is -1)
    !    Since we are overwriting the file (mode is FILE_MODE_WRITE), we are going to have to
    !    write the geometry (the DM), which can only be done on a brand new file.
    !

    ! Save the geometry to the file, erasing all previous content
    PetscCallA(DMView(dm,viewer,ierr))
    PetscCallA(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD,ierr))
    !
    !    Note how the exodus file is now open
    !
    ! "Format" the exodus result file, i.e. allocate space for nodal and zonal variables
    select case(sdim)
    case(2)
        numNodalVar = 3
        nodalVarName(1:numNodalVar) = ["U_x  ","U_y  ","Alpha"]
        numZonalVar = 3
        zonalVarName(1:numZonalVar) = ["Sigma_11","Sigma_22","Sigma_12"]
    case(3)
        numNodalVar = 4
        nodalVarName(1:numNodalVar) = ["U_x  ","U_y  ","U_z  ","Alpha"]
        numZonalVar = 6
        zonalVarName(1:numZonalVar) = ["Sigma_11","Sigma_22","Sigma_33","Sigma_23","Sigma_13","Sigma_12"]
    case default
        write(IOBuffer,'("No layout for dimension ",I2)') sdim
    end select
    PetscCallA(PetscViewerExodusIIGetId(viewer,exoid,ierr))
    PetscCallA(expvp(exoid, "E", numZonalVar,ierr))
    PetscCallA(expvan(exoid, "E", numZonalVar, zonalVarName,ierr))
    PetscCallA(expvp(exoid, "N", numNodalVar,ierr))
    PetscCallA(expvan(exoid, "N", numNodalVar, nodalVarName,ierr))
    PetscCallA(exinq(exoid, EX_INQ_ELEM_BLK,numCS,PETSC_NULL_REAL,sjunk,ierr))

    !    An exodusII truth table specifies which fields are saved at which time step
    !    It speeds up I/O but reserving space for fields in the file ahead of time.
    allocate(truthtable(numCS,numZonalVar))
    truthtable = .true.
    PetscCallA(expvtt(exoid, numCS, numZonalVar, truthtable, ierr))
    deallocate(truthtable)

    !   Writing time step information in the file. Note that this is currently broken in the exodus library for netcdf4 (HDF5-based) files */
    do step = 1,numstep
        PetscCallA(exptim(exoid,step,Real(step,kind=kPR),ierr))
    end do

    PetscCallA(DMSetUseNatural(dm,PETSC_TRUE,ierr))
    PetscCallA(DMPlexGetPartitioner(dm,part,ierr))
    PetscCallA(PetscPartitionerSetFromOptions(part,ierr))
    PetscCallA(DMPlexDistribute(dm,0_kPI,migrationSF,pdm,ierr))

    if (numProc > 1) then
        PetscCallA(DMPlexSetMigrationSF(pdm,migrationSF,ierr))
        PetscCallA(PetscSFDestroy(migrationSF,ierr))
        PetscCallA(DMDestroy(dm,ierr))
        dm = pdm
    end if
    PetscCallA(DMViewFromOptions(dm,PETSC_NULL_OPTIONS,"-dm_view",ierr))

    PetscCallA(PetscObjectGetComm(dm,comm,ierr))
    PetscCallA(PetscSectionCreate(comm, section,ierr))
    PetscCallA(PetscSectionSetNumFields(section, 3_kPI,ierr))
    PetscCallA(PetscSectionSetFieldName(section, fieldU, "U",ierr))
    PetscCallA(PetscSectionSetFieldName(section, fieldA, "Alpha",ierr))
    PetscCallA(PetscSectionSetFieldName(section, fieldS, "Sigma",ierr))
    PetscCallA(DMPlexGetChart(dm, pStart, pEnd,ierr))
    PetscCallA(PetscSectionSetChart(section, pStart, pEnd,ierr))

    allocate(pStartDepth(sdim+1))
    allocate(pEndDepth(sdim+1))
    do d = 1, sdim+1
        PetscCallA(DMPlexGetDepthStratum(dm, d-1, pStartDepth(d), pEndDepth(d),ierr))
    end do

    ! Vector field U, Scalar field Alpha, Tensor field Sigma
    PetscCallA(PetscSectionSetFieldComponents(section, fieldU, sdim,ierr))
    PetscCallA(PetscSectionSetFieldComponents(section, fieldA, 1_kPI,ierr))
    PetscCallA(PetscSectionSetFieldComponents(section, fieldS, sdim*(sdim+1)/2,ierr))

    ! Going through cell sets then cells, and setting up storage for the sections
    PetscCallA(DMGetLabelSize(dm, "Cell Sets", numCS, ierr))
    PetscCallA(DMGetLabelIdIS(dm, "Cell Sets", csIS, ierr))
    PetscCallA(ISGetIndicesF90(csIS, csID, ierr))
    do set = 1,numCS
        PetscCallA(DMGetStratumSize(dm, "Cell Sets", csID(set), numCells,ierr))
        PetscCallA(DMGetStratumIS(dm, "Cell Sets", csID(set), cellIS,ierr))
        if (numCells > 0) then
            select case(sdim)
            case(2)
                dofs => dofS2D
            case(3)
                dofs => dofS3D
            case default
                write(IOBuffer,'("No layout for dimension ",I2)') sdim
                SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,IOBuffer)
            end select ! sdim

            ! Identify cell type based on closure size only. This works for Tri/Tet/Quad/Hex meshes
            ! It will not be enough to identify more exotic elements like pyramid or prisms...  */
            PetscCallA(ISGetIndicesF90(cellIS, cellID,ierr))
            nullify(closureA)
            PetscCallA(DMPlexGetTransitiveClosure(dm,cellID(1), PETSC_TRUE, closureA,ierr))
            select case(size(closureA)/2)
            case(7) ! Tri
                if (order == 1) then
                    dofU => dofUP1Tri
                    dofA => dofAP1Tri
                else
                    dofU => dofUP2Tri
                    dofA => dofAP2Tri
                end if
            case(9) ! Quad
                if (order == 1) then
                    dofU => dofUP1Quad
                    dofA => dofAP1Quad
                else
                    dofU => dofUP2Quad
                    dofA => dofAP2Quad
                end if
            case(15) ! Tet
                if (order == 1) then
                    dofU => dofUP1Tet
                    dofA => dofAP1Tet
                else
                    dofU => dofUP2Tet
                    dofA => dofAP2Tet
                end if
            case(27) ! Hex
                if (order == 1) then
                    dofU => dofUP1Hex
                    dofA => dofAP1Hex
                else
                    dofU => dofUP2Hex
                    dofA => dofAP2Hex
                end if
            case default
                write(IOBuffer,'("Unknown element with closure size ",I2)') size(closureA)/2
                SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,IOBuffer)
            end select
            PetscCallA(DMPlexRestoreTransitiveClosure(dm, cellID(1), PETSC_TRUE,closureA,ierr))
            do cell = 1,numCells!
                nullify(closure)
                PetscCallA(DMPlexGetTransitiveClosure(dm, cellID(cell), PETSC_TRUE, closure,ierr))
                do p = 1,size(closure),2
                    ! find the depth of p
                    do d = 1,sdim+1
                        if ((closure(p) >= pStartDepth(d)) .and. (closure(p) < pEndDepth(d))) then
                            PetscCallA(PetscSectionSetDof(section, closure(p), dofU(d)+dofA(d)+dofS(d),ierr))
                            PetscCallA(PetscSectionSetFieldDof(section, closure(p), fieldU, dofU(d),ierr))
                            PetscCallA(PetscSectionSetFieldDof(section, closure(p), fieldA, dofA(d),ierr))
                            PetscCallA(PetscSectionSetFieldDof(section, closure(p), fieldS, dofS(d),ierr))
                        end if ! closure(p)
                    end do ! d
                end do ! p
                PetscCallA(DMPlexRestoreTransitiveClosure(dm, cellID(cell), PETSC_TRUE, closure,ierr))
            end do ! cell
            PetscCallA(ISRestoreIndicesF90(cellIS, cellID,ierr))
            PetscCallA(ISDestroy(cellIS,ierr))
        end if ! numCells
    end do ! set
    PetscCallA(ISRestoreIndicesF90(csIS, csID,ierr))
    PetscCallA(ISDestroy(csIS,ierr))
    PetscCallA(PetscSectionSetUp(section,ierr))
    PetscCallA(DMSetLocalSection(dm, section,ierr))
    PetscCallA(PetscObjectViewFromOptions(section, PETSC_NULL_SECTION, "-dm_section_view",ierr))
    PetscCallA(PetscSectionDestroy(section,ierr))

    ! Get DM and IS for each field of dm
    PetscCallA(DMCreateSubDM(dm, 1_kPI, fieldU,  isU,  dmU,ierr))
    PetscCallA(DMCreateSubDM(dm, 1_kPI, fieldA,  isA,  dmA,ierr))
    PetscCallA(DMCreateSubDM(dm, 1_kPI, fieldS,  isS,  dmS,ierr))
    PetscCallA(DMCreateSubDM(dm, 2_kPI, fieldUA, isUA, dmUA,ierr))

    !Create the exodus result file
    allocate(dmList(2))
    dmList(1) = dmU;
    dmList(2) = dmA;
    PetscCallA(DMCreateSuperDM(dmList,2_kPI,PETSC_NULL_IS,dmUA2,ierr))
    deallocate(dmList)

    PetscCallA(DMGetGlobalVector(dm,   X,ierr))
    PetscCallA(DMGetGlobalVector(dmU,  U,ierr))
    PetscCallA(DMGetGlobalVector(dmA,  A,ierr))
    PetscCallA(DMGetGlobalVector(dmS,  S,ierr))
    PetscCallA(DMGetGlobalVector(dmUA, UA,ierr))
    PetscCallA(DMGetGlobalVector(dmUA2, UA2,ierr))

    PetscCallA(PetscObjectSetName(U,  "U",ierr))
    PetscCallA(PetscObjectSetName(A,  "Alpha",ierr))
    PetscCallA(PetscObjectSetName(S,  "Sigma",ierr))
    PetscCallA(PetscObjectSetName(UA, "UAlpha",ierr))
    PetscCallA(PetscObjectSetName(UA2, "UAlpha2",ierr))
    PetscCallA(VecSet(X, -111.0_kPR,ierr))

    ! Setting u to [x,y,z]  and alpha to x^2+y^2+z^2 by writing in UAlpha then restricting to U and Alpha */
    PetscCallA(DMGetLocalSection(dmUA, sectionUA,ierr))
    PetscCallA(DMGetLocalVector(dmUA, UALoc,ierr))
    PetscCallA(VecGetArrayF90(UALoc, cval,ierr))
    PetscCallA(DMGetCoordinateSection(dmUA, coordSection,ierr))
    PetscCallA(DMGetCoordinatesLocal(dmUA, coord,ierr))
    PetscCallA(DMPlexGetChart(dmUA, pStart, pEnd,ierr))

    do p = pStart,pEnd-1
        PetscCallA(PetscSectionGetDof(sectionUA, p, dofUA,ierr))
        if (dofUA > 0) then
            PetscCallA(PetscSectionGetOffset(sectionUA, p, offUA,ierr))
            PetscCallA(DMPlexVecGetClosure(dmUA, coordSection, coord, p, xyz,ierr))
            closureSize = size(xyz)
            do i = 1,sdim
                do j = 0, closureSize-1,sdim
                    cval(offUA+i) = cval(offUA+i) + xyz(j/sdim+i)
                end do
                cval(offUA+i) = cval(offUA+i) * sdim / closureSize;
                cval(offUA+sdim+1) = cval(offUA+sdim+1) + cval(offUA+i)**2
            end do
            PetscCallA(DMPlexVecRestoreClosure(dmUA, coordSection, coord, p, xyz,ierr))
        end if
    end do

    PetscCallA(VecRestoreArrayF90(UALoc, cval,ierr))
    PetscCallA(DMLocalToGlobalBegin(dmUA, UALoc, INSERT_VALUES, UA,ierr))
    PetscCallA(DMLocalToGlobalEnd(dmUA, UALoc, INSERT_VALUES, UA,ierr))
    PetscCallA(DMRestoreLocalVector(dmUA, UALoc,ierr))

    !Update X
    PetscCallA(VecISCopy(X, isUA, SCATTER_FORWARD, UA,ierr))
    ! Restrict to U and Alpha
    PetscCallA(VecISCopy(X, isU, SCATTER_REVERSE, U,ierr))
    PetscCallA(VecISCopy(X, isA, SCATTER_REVERSE, A,ierr))
    PetscCallA(VecViewFromOptions(UA, PETSC_NULL_OPTIONS, "-ua_vec_view",ierr))
    PetscCallA(VecViewFromOptions(U, PETSC_NULL_OPTIONS, "-u_vec_view",ierr))
    PetscCallA(VecViewFromOptions(A, PETSC_NULL_OPTIONS, "-a_vec_view",ierr))
    ! restrict to UA2
    PetscCallA(VecISCopy(X, isUA, SCATTER_REVERSE, UA2,ierr))
    PetscCallA(VecViewFromOptions(UA2, PETSC_NULL_OPTIONS, "-ua2_vec_view",ierr))

    ! Writing nodal variables to ExodusII file
    PetscCallA(DMSetOutputSequenceNumber(dmU,0_kPI,time,ierr))
    PetscCallA(DMSetOutputSequenceNumber(dmA,0_kPI,time,ierr))

    PetscCallA(VecView(U, viewer,ierr))
    PetscCallA(VecView(A, viewer,ierr))

    ! Saving U and Alpha in one shot.
    ! For this, we need to cheat and change the Vec's name
    ! Note that in the end we write variables one component at a time,
    ! so that there is no real value in doing this

    PetscCallA(DMSetOutputSequenceNumber(dmUA,1_kPI,time,ierr))
    PetscCallA(DMGetGlobalVector(dmUA, tmpVec,ierr))
    PetscCallA(VecCopy(UA, tmpVec,ierr))
    PetscCallA(PetscObjectSetName(tmpVec, "U",ierr))
    PetscCallA(VecView(tmpVec, viewer,ierr))
    ! Reading nodal variables in Exodus file
    PetscCallA(VecSet(tmpVec, -1000.0_kPR,ierr))
    PetscCallA(VecLoad(tmpVec, viewer,ierr))
    PetscCallA(VecAXPY(UA, -1.0_kPR, tmpVec,ierr))
    PetscCallA(VecNorm(UA, NORM_INFINITY, norm,ierr))
    if (norm > PETSC_SQRT_MACHINE_EPSILON) then
        write(IOBuffer,'("UAlpha ||Vin - Vout|| = ",ES12.5)') norm
    end if
    PetscCallA(DMRestoreGlobalVector(dmUA, tmpVec,ierr))

    ! ! same thing with the UA2 Vec obtained from the superDM
    PetscCallA(DMGetGlobalVector(dmUA2, tmpVec,ierr))
    PetscCallA(VecCopy(UA2, tmpVec,ierr))
    PetscCallA(PetscObjectSetName(tmpVec, "U",ierr))
    PetscCallA(DMSetOutputSequenceNumber(dmUA2,2_kPI,time,ierr))
    PetscCallA(VecView(tmpVec, viewer,ierr))
    ! Reading nodal variables in Exodus file
    PetscCallA(VecSet(tmpVec, -1000.0_kPR,ierr))
    PetscCallA(VecLoad(tmpVec,viewer,ierr))
    PetscCallA(VecAXPY(UA2, -1.0_kPR, tmpVec,ierr))
    PetscCallA(VecNorm(UA2, NORM_INFINITY, norm,ierr))
    if (norm > PETSC_SQRT_MACHINE_EPSILON) then
        write(IOBuffer,'("UAlpha2 ||Vin - Vout|| = ",ES12.5)') norm
    end if
    PetscCallA(DMRestoreGlobalVector(dmUA2, tmpVec,ierr))

    ! Building and saving Sigma
    !   We set sigma_0 = rank (to see partitioning)
    !          sigma_1 = cell set ID
    !          sigma_2 = x_coordinate of the cell center of mass
    PetscCallA(DMGetCoordinateSection(dmS, coordSection,ierr))
    PetscCallA(DMGetCoordinatesLocal(dmS, coord,ierr))
    PetscCallA(DMGetLabelIdIS(dmS, "Cell Sets", csIS,ierr))
    PetscCallA(DMGetLabelSize(dmS, "Cell Sets",numCS,ierr))
    PetscCallA(ISGetIndicesF90(csIS, csID,ierr))

    do set = 1, numCS
        PetscCallA(DMGetStratumIS(dmS, "Cell Sets", csID(set), cellIS,ierr))
        PetscCallA(ISGetIndicesF90(cellIS, cellID,ierr))
        PetscCallA(ISGetSize(cellIS, numCells,ierr))
        do cell = 1,numCells
            PetscCallA(DMPlexVecGetClosure(dmS, PETSC_NULL_SECTION, S, cellID(cell), cval,ierr))
            PetscCallA(DMPlexVecGetClosure(dmS, coordSection, coord, cellID(cell), xyz,ierr))
            cval(1) = rank
            cval(2) = csID(set)
            cval(3) = 0.0_kPR
            do c = 1, size(xyz),sdim
                cval(3) = cval(3) + xyz(c)
            end do
            cval(3) = cval(3) * sdim / size(xyz)
            PetscCallA(DMPlexVecSetClosure(dmS, PETSC_NULL_SECTION, S, cellID(cell), cval, INSERT_ALL_VALUES,ierr))
            PetscCallA(DMPlexVecRestoreClosure(dmS, PETSC_NULL_SECTION, S, cellID(cell), cval,ierr))
            PetscCallA(DMPlexVecRestoreClosure(dmS, coordSection, coord, cellID(cell), xyz,ierr))
        end do
        PetscCallA(ISRestoreIndicesF90(cellIS, cellID,ierr))
        PetscCallA(ISDestroy(cellIS,ierr))
    end do
    PetscCallA(ISRestoreIndicesF90(csIS, csID,ierr))
    PetscCallA(ISDestroy(csIS,ierr))
    PetscCallA(VecViewFromOptions(S, PETSC_NULL_OPTIONS, "-s_vec_view",ierr))

    ! Writing zonal variables in Exodus file
    PetscCallA(DMSetOutputSequenceNumber(dmS,0_kPI,time,ierr))
    PetscCallA(VecView(S,viewer,ierr))

    ! Reading zonal variables in Exodus file */
    PetscCallA(DMGetGlobalVector(dmS, tmpVec,ierr))
    PetscCallA(VecSet(tmpVec, -1000.0_kPR,ierr))
    PetscCallA(PetscObjectSetName(tmpVec, "Sigma",ierr))
    PetscCallA(VecLoad(tmpVec,viewer,ierr))
    PetscCallA(VecAXPY(S, -1.0_kPR, tmpVec,ierr))
    PetscCallA(VecNorm(S, NORM_INFINITY,norm,ierr))
    if (norm > PETSC_SQRT_MACHINE_EPSILON) then
       write(IOBuffer,'("Sigma ||Vin - Vout|| = ",ES12.5)') norm
    end if
    PetscCallA(DMRestoreGlobalVector(dmS, tmpVec,ierr))

    PetscCallA(DMRestoreGlobalVector(dmUA2, UA2,ierr))
    PetscCallA(DMRestoreGlobalVector(dmUA, UA,ierr))
    PetscCallA(DMRestoreGlobalVector(dmS,  S,ierr))
    PetscCallA(DMRestoreGlobalVector(dmA,  A,ierr))
    PetscCallA(DMRestoreGlobalVector(dmU,  U,ierr))
    PetscCallA(DMRestoreGlobalVector(dm,   X,ierr))
    PetscCallA(DMDestroy(dmU,ierr))
    PetscCallA(ISDestroy(isU,ierr))
    PetscCallA(DMDestroy(dmA,ierr))
    PetscCallA(ISDestroy(isA,ierr))
    PetscCallA(DMDestroy(dmS,ierr))
    PetscCallA(ISDestroy(isS,ierr))
    PetscCallA(DMDestroy(dmUA,ierr))
    PetscCallA(ISDestroy(isUA,ierr))
    PetscCallA(DMDestroy(dmUA2,ierr))
    PetscCallA(DMDestroy(dm,ierr))

    deallocate(pStartDepth)
    deallocate(pEndDepth)

    PetscCallA(PetscViewerDestroy(viewer,ierr))
    PetscCallA(PetscFinalize(ierr))
end program ex62f90

! /*TEST
!
! build:
!   requires: exodusii pnetcdf !complex
!   # 2D seq
! test:
!   suffix: 0
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
!   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! test:
!   suffix: 1
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
!
! test:
!   suffix: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
!   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! test:
!   suffix: 3
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! test:
!   suffix: 4
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! test:
!   suffix: 5
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
!   # 2D par
! test:
!   suffix: 6
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
!   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! test:
!   suffix: 7
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! test:
!   suffix: 8
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
!   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: invalid dimension ID or name
! test:
!   suffix: 9
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! test:
!   suffix: 10
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! test:
!   # Something is now broken with parallel read/write for wahtever shape H is
!   TODO: broken
!   suffix: 11
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2

!   #3d seq
! test:
!   suffix: 12
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! test:
!   suffix: 13
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
!   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! test:
!   suffix: 14
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! test:
!   suffix: 15
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
!   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
!   #3d par
! test:
!   suffix: 16
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! test:
!   suffix: 17
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
!   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! test:
!   suffix: 18
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! test:
!   suffix: 19
!   nsize: 2
!   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
!   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! TEST*/
