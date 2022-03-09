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

    call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
    if (ierr /= 0) then
      print*,'Unable to initialize PETSc'
      stop
    endif

    call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
    call MPI_Comm_size(PETSC_COMM_WORLD,numProc,ierr)
    call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-i",ifilename,flg,ierr);CHKERRA(ierr)
    if (.not. flg) then
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"missing input file name -i <input file name>")
    end if
    call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-o",ofilename,flg,ierr);CHKERRA(ierr)
    if (.not. flg) then
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"missing output file name -o <output file name>")
    end if
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-order",order,flg,ierr);CHKERRA(ierr)
    if ((order > 2) .or. (order < 1)) then
        write(IOBuffer,'("Unsupported polynomial order ", I2, " not in [1,2]")') order
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,IOBuffer)
    end if

    ! Read the mesh in any supported format
    call DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename,PETSC_NULL_CHARACTER,PETSC_TRUE,dm,ierr);CHKERRA(ierr)
    call DMPlexDistributeSetDefault(dm,PETSC_FALSE,ierr);CHKERRA(ierr);
    call DMSetFromOptions(dm,ierr);CHKERRA(ierr);
    call DMGetDimension(dm, sdim,ierr);CHKERRA(ierr)
    call DMViewFromOptions(dm, PETSC_NULL_OPTIONS,"-dm_view",ierr);CHKERRA(ierr);

    ! Create the exodus result file

    ! enable exodus debugging information
    call exopts(EXVRBS+EXDEBG,ierr)
    ! Create the exodus file
    call PetscViewerExodusIIOpen(PETSC_COMM_WORLD,ofilename,FILE_MODE_WRITE,viewer,ierr);CHKERRA(ierr)
    ! The long way would be
    !
    ! call PetscViewerCreate(PETSC_COMM_WORLD,viewer,ierr);CHKERRA(ierr)
    ! call PetscViewerSetType(viewer,PETSCVIEWEREXODUSII,ierr);CHKERRA(ierr)
    ! call PetscViewerFileSetMode(viewer,FILE_MODE_WRITE,ierr);CHKERRA(ierr)
    ! call PetscViewerFileSetName(viewer,ofilename,ierr);CHKERRA(ierr)

    ! set the mesh order
    call PetscViewerExodusIISetOrder(viewer,order,ierr);CHKERRA(ierr)
    call PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
    !
    !    Notice how the exodus file is actually NOT open at this point (exoid is -1)
    !    Since we are overwriting the file (mode is FILE_MODE_WRITE), we are going to have to
    !    write the geometry (the DM), which can only be done on a brand new file.
    !

    ! Save the geometry to the file, erasing all previous content
    call DMView(dm,viewer,ierr);CHKERRA(ierr)
    call PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
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
    call PetscViewerExodusIIGetId(viewer,exoid,ierr);CHKERRA(ierr)
    call expvp(exoid, "E", numZonalVar,ierr)
    call expvan(exoid, "E", numZonalVar, zonalVarName,ierr)
    call expvp(exoid, "N", numNodalVar,ierr)
    call expvan(exoid, "N", numNodalVar, nodalVarName,ierr)
    call exinq(exoid, EX_INQ_ELEM_BLK,numCS,PETSC_NULL_REAL,sjunk,ierr)

    !    An exodusII truth table specifies which fields are saved at which time step
    !    It speeds up I/O but reserving space for fields in the file ahead of time.
    allocate(truthtable(numCS,numZonalVar))
    truthtable = .true.
    call expvtt(exoid, numCS, numZonalVar, truthtable, ierr)
    deallocate(truthtable)

    !   Writing time step information in the file. Note that this is currently broken in the exodus library for netcdf4 (HDF5-based) files */
    do step = 1,numstep
        call exptim(exoid,step,Real(step,kind=kPR),ierr)
    end do

    call DMSetUseNatural(dm,PETSC_TRUE,ierr);CHKERRA(ierr)
    call DMPlexGetPartitioner(dm,part,ierr);CHKERRA(ierr)
    call PetscPartitionerSetFromOptions(part,ierr);CHKERRA(ierr)
    call DMPlexDistribute(dm,0_kPI,migrationSF,pdm,ierr);CHKERRA(ierr)

    if (numProc > 1) then
        call DMPlexSetMigrationSF(pdm,migrationSF,ierr);CHKERRA(ierr)
        call PetscSFDestroy(migrationSF,ierr);CHKERRA(ierr)
        call DMDestroy(dm,ierr);CHKERRA(ierr)
        dm = pdm
    end if
    call DMViewFromOptions(dm,PETSC_NULL_OPTIONS,"-dm_view",ierr);CHKERRA(ierr)

    call PetscObjectGetComm(dm,comm,ierr);CHKERRA(ierr)
    call PetscSectionCreate(comm, section,ierr);CHKERRA(ierr)
    call PetscSectionSetNumFields(section, 3_kPI,ierr);CHKERRA(ierr)
    call PetscSectionSetFieldName(section, fieldU, "U",ierr);CHKERRA(ierr)
    call PetscSectionSetFieldName(section, fieldA, "Alpha",ierr);CHKERRA(ierr)
    call PetscSectionSetFieldName(section, fieldS, "Sigma",ierr);CHKERRA(ierr)
    call DMPlexGetChart(dm, pStart, pEnd,ierr);CHKERRA(ierr)
    call PetscSectionSetChart(section, pStart, pEnd,ierr);CHKERRA(ierr)

    allocate(pStartDepth(sdim+1))
    allocate(pEndDepth(sdim+1))
    do d = 1, sdim+1
        call DMPlexGetDepthStratum(dm, d-1, pStartDepth(d), pEndDepth(d),ierr);CHKERRA(ierr)
    end do

    ! Vector field U, Scalar field Alpha, Tensor field Sigma
    call PetscSectionSetFieldComponents(section, fieldU, sdim,ierr);CHKERRA(ierr);
    call PetscSectionSetFieldComponents(section, fieldA, 1_kPI,ierr);CHKERRA(ierr);
    call PetscSectionSetFieldComponents(section, fieldS, sdim*(sdim+1)/2,ierr);CHKERRA(ierr);

    ! Going through cell sets then cells, and setting up storage for the sections
    call DMGetLabelSize(dm, "Cell Sets", numCS, ierr);CHKERRA(ierr)
    call DMGetLabelIdIS(dm, "Cell Sets", csIS, ierr);CHKERRA(ierr)
    call ISGetIndicesF90(csIS, csID, ierr);CHKERRA(ierr)
    do set = 1,numCS
        call DMGetStratumSize(dm, "Cell Sets", csID(set), numCells,ierr);CHKERRA(ierr)
        call DMGetStratumIS(dm, "Cell Sets", csID(set), cellIS,ierr);CHKERRA(ierr)
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
            call ISGetIndicesF90(cellIS, cellID,ierr);CHKERRA(ierr)
            nullify(closureA)
            call DMPlexGetTransitiveClosure(dm,cellID(1), PETSC_TRUE, closureA,ierr);CHKERRA(ierr)
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
            call DMPlexRestoreTransitiveClosure(dm, cellID(1), PETSC_TRUE,closureA,ierr);CHKERRA(ierr)
            do cell = 1,numCells!
                nullify(closure)
                call DMPlexGetTransitiveClosure(dm, cellID(cell), PETSC_TRUE, closure,ierr);CHKERRA(ierr)
                do p = 1,size(closure),2
                    ! find the depth of p
                    do d = 1,sdim+1
                        if ((closure(p) >= pStartDepth(d)) .and. (closure(p) < pEndDepth(d))) then
                            call PetscSectionSetDof(section, closure(p), dofU(d)+dofA(d)+dofS(d),ierr);CHKERRA(ierr)
                            call PetscSectionSetFieldDof(section, closure(p), fieldU, dofU(d),ierr);CHKERRA(ierr)
                            call PetscSectionSetFieldDof(section, closure(p), fieldA, dofA(d),ierr);CHKERRA(ierr)
                            call PetscSectionSetFieldDof(section, closure(p), fieldS, dofS(d),ierr);CHKERRA(ierr)
                        end if ! closure(p)
                    end do ! d
                end do ! p
                call DMPlexRestoreTransitiveClosure(dm, cellID(cell), PETSC_TRUE, closure,ierr);CHKERRA(ierr)
            end do ! cell
            call ISRestoreIndicesF90(cellIS, cellID,ierr);CHKERRA(ierr)
            call ISDestroy(cellIS,ierr);CHKERRA(ierr)
        end if ! numCells
    end do ! set
    call ISRestoreIndicesF90(csIS, csID,ierr);CHKERRA(ierr)
    call ISDestroy(csIS,ierr);CHKERRA(ierr)
    call PetscSectionSetUp(section,ierr);CHKERRA(ierr)
    call DMSetLocalSection(dm, section,ierr);CHKERRA(ierr)
    call PetscObjectViewFromOptions(section, PETSC_NULL_SECTION, "-dm_section_view",ierr);CHKERRQ(ierr)
    call PetscSectionDestroy(section,ierr);CHKERRA(ierr)

    ! Get DM and IS for each field of dm
    call DMCreateSubDM(dm, 1_kPI, fieldU,  isU,  dmU,ierr);CHKERRA(ierr)
    call DMCreateSubDM(dm, 1_kPI, fieldA,  isA,  dmA,ierr);CHKERRA(ierr)
    call DMCreateSubDM(dm, 1_kPI, fieldS,  isS,  dmS,ierr);CHKERRA(ierr)
    call DMCreateSubDM(dm, 2_kPI, fieldUA, isUA, dmUA,ierr);CHKERRA(ierr)

    !Create the exodus result file
    allocate(dmList(2))
    dmList(1) = dmU;
    dmList(2) = dmA;
    call DMCreateSuperDM(dmList,2_kPI,PETSC_NULL_IS,dmUA2,ierr);CHKERRA(ierr)
    deallocate(dmList)

    call DMGetGlobalVector(dm,   X,ierr);CHKERRA(ierr)
    call DMGetGlobalVector(dmU,  U,ierr);CHKERRA(ierr)
    call DMGetGlobalVector(dmA,  A,ierr);CHKERRA(ierr)
    call DMGetGlobalVector(dmS,  S,ierr);CHKERRA(ierr)
    call DMGetGlobalVector(dmUA, UA,ierr);CHKERRA(ierr)
    call DMGetGlobalVector(dmUA2, UA2,ierr);CHKERRA(ierr)

    call PetscObjectSetName(U,  "U",ierr);CHKERRA(ierr)
    call PetscObjectSetName(A,  "Alpha",ierr);CHKERRA(ierr)
    call PetscObjectSetName(S,  "Sigma",ierr);CHKERRA(ierr)
    call PetscObjectSetName(UA, "UAlpha",ierr);CHKERRA(ierr)
    call PetscObjectSetName(UA2, "UAlpha2",ierr);CHKERRA(ierr)
    call VecSet(X, -111.0_kPR,ierr);CHKERRA(ierr)

    ! Setting u to [x,y,z]  and alpha to x^2+y^2+z^2 by writing in UAlpha then restricting to U and Alpha */
    call DMGetLocalSection(dmUA, sectionUA,ierr);CHKERRA(ierr)
    call DMGetLocalVector(dmUA, UALoc,ierr);CHKERRA(ierr)
    call VecGetArrayF90(UALoc, cval,ierr);CHKERRA(ierr)
    call DMGetCoordinateSection(dmUA, coordSection,ierr);CHKERRA(ierr)
    call DMGetCoordinatesLocal(dmUA, coord,ierr);CHKERRA(ierr)
    call DMPlexGetChart(dmUA, pStart, pEnd,ierr);CHKERRA(ierr)

    do p = pStart,pEnd-1
        call PetscSectionGetDof(sectionUA, p, dofUA,ierr);CHKERRA(ierr)
        if (dofUA > 0) then
            call PetscSectionGetOffset(sectionUA, p, offUA,ierr);CHKERRA(ierr)
            call DMPlexVecGetClosure(dmUA, coordSection, coord, p, xyz,ierr);CHKERRA(ierr)
            closureSize = size(xyz)
            do i = 1,sdim
                do j = 0, closureSize-1,sdim
                    cval(offUA+i) = cval(offUA+i) + xyz(j/sdim+i)
                end do
                cval(offUA+i) = cval(offUA+i) * sdim / closureSize;
                cval(offUA+sdim+1) = cval(offUA+sdim+1) + cval(offUA+i)**2
            end do
            call DMPlexVecRestoreClosure(dmUA, coordSection, coord, p, xyz,ierr);CHKERRA(ierr)
        end if
    end do

    call VecRestoreArrayF90(UALoc, cval,ierr);CHKERRA(ierr)
    call DMLocalToGlobalBegin(dmUA, UALoc, INSERT_VALUES, UA,ierr);CHKERRA(ierr)
    call DMLocalToGlobalEnd(dmUA, UALoc, INSERT_VALUES, UA,ierr);CHKERRA(ierr)
    call DMRestoreLocalVector(dmUA, UALoc,ierr);CHKERRA(ierr)

    !Update X
    call VecISCopy(X, isUA, SCATTER_FORWARD, UA,ierr);CHKERRA(ierr)
    ! Restrict to U and Alpha
    call VecISCopy(X, isU, SCATTER_REVERSE, U,ierr);CHKERRA(ierr)
    call VecISCopy(X, isA, SCATTER_REVERSE, A,ierr);CHKERRA(ierr)
    call VecViewFromOptions(UA, PETSC_NULL_OPTIONS, "-ua_vec_view",ierr);CHKERRA(ierr)
    call VecViewFromOptions(U, PETSC_NULL_OPTIONS, "-u_vec_view",ierr);CHKERRA(ierr)
    call VecViewFromOptions(A, PETSC_NULL_OPTIONS, "-a_vec_view",ierr);CHKERRA(ierr)
    ! restrict to UA2
    call VecISCopy(X, isUA, SCATTER_REVERSE, UA2,ierr);CHKERRA(ierr)
    call VecViewFromOptions(UA2, PETSC_NULL_OPTIONS, "-ua2_vec_view",ierr);CHKERRA(ierr)

    ! Writing nodal variables to ExodusII file
    call DMSetOutputSequenceNumber(dmU,0_kPI,time,ierr);CHKERRA(ierr)
    call DMSetOutputSequenceNumber(dmA,0_kPI,time,ierr);CHKERRA(ierr)

    call VecView(U, viewer,ierr);CHKERRA(ierr)
    call VecView(A, viewer,ierr);CHKERRA(ierr)

    ! Saving U and Alpha in one shot.
    ! For this, we need to cheat and change the Vec's name
    ! Note that in the end we write variables one component at a time,
    ! so that there is no real value in doing this

    call DMSetOutputSequenceNumber(dmUA,1_kPI,time,ierr);CHKERRA(ierr)
    call DMGetGlobalVector(dmUA, tmpVec,ierr);CHKERRA(ierr)
    call VecCopy(UA, tmpVec,ierr);CHKERRA(ierr)
    call PetscObjectSetName(tmpVec, "U",ierr);CHKERRA(ierr)
    call VecView(tmpVec, viewer,ierr);CHKERRA(ierr)
    ! Reading nodal variables in Exodus file
    call VecSet(tmpVec, -1000.0_kPR,ierr);CHKERRA(ierr)
    call VecLoad(tmpVec, viewer,ierr);CHKERRA(ierr)
    call VecAXPY(UA, -1.0_kPR, tmpVec,ierr);CHKERRA(ierr)
    call VecNorm(UA, NORM_INFINITY, norm,ierr);CHKERRA(ierr)
    if (norm > PETSC_SQRT_MACHINE_EPSILON) then
        write(IOBuffer,'("UAlpha ||Vin - Vout|| = ",ES12.5)') norm
    end if
    call DMRestoreGlobalVector(dmUA, tmpVec,ierr);CHKERRA(ierr)

    ! ! same thing with the UA2 Vec obtained from the superDM
    call DMGetGlobalVector(dmUA2, tmpVec,ierr);CHKERRA(ierr)
    call VecCopy(UA2, tmpVec,ierr);CHKERRA(ierr)
    call PetscObjectSetName(tmpVec, "U",ierr);CHKERRA(ierr)
    call DMSetOutputSequenceNumber(dmUA2,2_kPI,time,ierr);CHKERRA(ierr)
    call VecView(tmpVec, viewer,ierr);CHKERRA(ierr)
    ! Reading nodal variables in Exodus file
    call VecSet(tmpVec, -1000.0_kPR,ierr);CHKERRA(ierr)
    call VecLoad(tmpVec,viewer,ierr);CHKERRA(ierr)
    call VecAXPY(UA2, -1.0_kPR, tmpVec,ierr);CHKERRA(ierr)
    call VecNorm(UA2, NORM_INFINITY, norm,ierr);CHKERRA(ierr)
    if (norm > PETSC_SQRT_MACHINE_EPSILON) then
        write(IOBuffer,'("UAlpha2 ||Vin - Vout|| = ",ES12.5)') norm
    end if
    call DMRestoreGlobalVector(dmUA2, tmpVec,ierr);CHKERRA(ierr)

    ! Building and saving Sigma
    !   We set sigma_0 = rank (to see partitioning)
    !          sigma_1 = cell set ID
    !          sigma_2 = x_coordinate of the cell center of mass
    call DMGetCoordinateSection(dmS, coordSection,ierr);CHKERRA(ierr)
    call DMGetCoordinatesLocal(dmS, coord,ierr);CHKERRA(ierr)
    call DMGetLabelIdIS(dmS, "Cell Sets", csIS,ierr);CHKERRA(ierr)
    call DMGetLabelSize(dmS, "Cell Sets",numCS,ierr);CHKERRA(ierr)
    call ISGetIndicesF90(csIS, csID,ierr);CHKERRA(ierr)

    do set = 1, numCS
        call DMGetStratumIS(dmS, "Cell Sets", csID(set), cellIS,ierr);CHKERRA(ierr)
        call ISGetIndicesF90(cellIS, cellID,ierr);CHKERRA(ierr)
        call ISGetSize(cellIS, numCells,ierr);CHKERRA(ierr)
        do cell = 1,numCells
            call DMPlexVecGetClosure(dmS, PETSC_NULL_SECTION, S, cellID(cell), cval,ierr);CHKERRA(ierr)
            call DMPlexVecGetClosure(dmS, coordSection, coord, cellID(cell), xyz,ierr);CHKERRA(ierr)
            cval(1) = rank
            cval(2) = csID(set)
            cval(3) = 0.0_kPR
            do c = 1, size(xyz),sdim
                cval(3) = cval(3) + xyz(c)
            end do
            cval(3) = cval(3) * sdim / size(xyz)
            call DMPlexVecSetClosure(dmS, PETSC_NULL_SECTION, S, cellID(cell), cval, INSERT_ALL_VALUES,ierr);CHKERRA(ierr)
            call DMPlexVecRestoreClosure(dmS, PETSC_NULL_SECTION, S, cellID(cell), cval,ierr);CHKERRA(ierr)
            call DMPlexVecRestoreClosure(dmS, coordSection, coord, cellID(cell), xyz,ierr);CHKERRA(ierr)
        end do
        call ISRestoreIndicesF90(cellIS, cellID,ierr);CHKERRA(ierr)
        call ISDestroy(cellIS,ierr);CHKERRA(ierr)
    end do
    call ISRestoreIndicesF90(csIS, csID,ierr);CHKERRA(ierr)
    call ISDestroy(csIS,ierr);CHKERRA(ierr)
    call VecViewFromOptions(S, PETSC_NULL_OPTIONS, "-s_vec_view",ierr);CHKERRA(ierr)

    ! Writing zonal variables in Exodus file
    call DMSetOutputSequenceNumber(dmS,0_kPI,time,ierr);CHKERRA(ierr)
    call VecView(S,viewer,ierr);CHKERRA(ierr)

    ! Reading zonal variables in Exodus file */
    call DMGetGlobalVector(dmS, tmpVec,ierr);CHKERRA(ierr)
    call VecSet(tmpVec, -1000.0_kPR,ierr);CHKERRA(ierr)
    call PetscObjectSetName(tmpVec, "Sigma",ierr);CHKERRA(ierr)
    call VecLoad(tmpVec,viewer,ierr);CHKERRA(ierr)
    call VecAXPY(S, -1.0_kPR, tmpVec,ierr);CHKERRA(ierr)
    call VecNorm(S, NORM_INFINITY,norm,ierr);CHKERRQ(ierr)
    if (norm > PETSC_SQRT_MACHINE_EPSILON) then
       write(IOBuffer,'("Sigma ||Vin - Vout|| = ",ES12.5)') norm
    end if
    call DMRestoreGlobalVector(dmS, tmpVec,ierr);CHKERRA(ierr)

    call DMRestoreGlobalVector(dmUA2, UA2,ierr);CHKERRA(ierr)
    call DMRestoreGlobalVector(dmUA, UA,ierr);CHKERRA(ierr)
    call DMRestoreGlobalVector(dmS,  S,ierr);CHKERRA(ierr)
    call DMRestoreGlobalVector(dmA,  A,ierr);CHKERRA(ierr)
    call DMRestoreGlobalVector(dmU,  U,ierr);CHKERRA(ierr)
    call DMRestoreGlobalVector(dm,   X,ierr);CHKERRA(ierr)
    call DMDestroy(dmU,ierr);CHKERRA(ierr);
    call ISDestroy(isU,ierr);CHKERRA(ierr)
    call DMDestroy(dmA,ierr);CHKERRA(ierr);
    call ISDestroy(isA,ierr);CHKERRA(ierr)
    call DMDestroy(dmS,ierr);CHKERRA(ierr);
    call ISDestroy(isS,ierr);CHKERRA(ierr)
    call DMDestroy(dmUA,ierr);CHKERRA(ierr)
    call ISDestroy(isUA,ierr);CHKERRA(ierr)
    call DMDestroy(dmUA2,ierr);CHKERRA(ierr)
    call DMDestroy(dm,ierr);CHKERRA(ierr)

    deallocate(pStartDepth)
    deallocate(pEndDepth)

    call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
    call PetscFinalize(ierr)
end program ex62f90

! /*TEST
!
! build:
!   requires: exodusii pnetcdf !complex
!   # 2D seq
! # test:
! #   suffix: 0
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! #   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! # test:
! #   suffix: 1
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! #
! # test:
! #   suffix: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! #   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! # test:
! #   suffix: 3
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! # test:
! #   suffix: 4
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! # test:
! #   suffix: 5
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! #   # 2D par
! # test:
! #   suffix: 6
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! #   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! # test:
! #   suffix: 7
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! # test:
! #   suffix: 8
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! #   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: invalid dimension ID or name
! # test:
! #   suffix: 9
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! # test:
! #   suffix: 10
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! # test:
! #   # Something is now broken with parallel read/write for wahtever shape H is
! #   TODO: broken
! #   suffix: 11
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! #
! #   #3d seq
! # test:
! #   suffix: 12
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! # test:
! #   suffix: 13
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! #   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! # test:
! #   suffix: 14
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! # test:
! #   suffix: 15
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! #   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! #   #3d par
! # test:
! #   suffix: 16
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! # test:
! #   suffix: 17
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
! #   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! # test:
! #   suffix: 18
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! # test:
! #   suffix: 19
! #   nsize: 2
! #   args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
! #   #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
! TEST*/
