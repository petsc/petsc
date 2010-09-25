cdef extern from "petscda.h" nogil:

    ctypedef enum PetscDAStencilType "DAStencilType":
        DA_STENCIL_STAR
        DA_STENCIL_BOX

    ctypedef enum PetscDAPeriodicType "DAPeriodicType":
        DA_PERIODIC_NONE  "DA_NONPERIODIC"
        DA_PERIODIC_X     "DA_XPERIODIC"
        DA_PERIODIC_Y     "DA_YPERIODIC"
        DA_PERIODIC_Z     "DA_ZPERIODIC"
        DA_PERIODIC_XY    "DA_XYPERIODIC"
        DA_PERIODIC_XZ    "DA_XZPERIODIC"
        DA_PERIODIC_YZ    "DA_YZPERIODIC"
        DA_PERIODIC_XYZ   "DA_XYZPERIODIC"
        DA_GHOSTED_XYZ    "DA_XYZGHOSTED"

    ctypedef enum PetscDAInterpolationType "DAInterpolationType":
        DA_INTERPOLATION_Q0 "DA_Q0"
        DA_INTERPOLATION_Q1 "DA_Q1"

    ctypedef enum PetscDAElementType "DAElementType":
        DA_ELEMENT_P1
        DA_ELEMENT_Q1

    int DAView(PetscDA,PetscViewer)
    int DADestroy(PetscDA)

    int DACreateND(MPI_Comm,
                   PetscInt,PetscInt,                # dim, dof
                   PetscInt,PetscInt,PetscInt,       # M, N, P
                   PetscInt,PetscInt,PetscInt,       # m, n, p
                   PetscInt[],PetscInt[],PetscInt[], # lx, ly, lz
                   PetscDAPeriodicType,              # periodicity
                   PetscDAStencilType,               # stencil type
                   PetscInt,                         # stencil width
                   PetscDA*)

    int DASetOptionsPrefix(PetscDA,char[])
    int DASetFromOptions(PetscDA)
    int DASetElementType(PetscDA,PetscDAElementType)
    int DASetInterpolationType(PetscDA,PetscDAInterpolationType)
    int DAGetInterpolation(PetscDA,PetscDA,PetscMat*,PetscVec*)
    int DAGetInfo(PetscDA,
                  PetscInt*,
                  PetscInt*,PetscInt*,PetscInt*,
                  PetscInt*,PetscInt*,PetscInt*,
                  PetscInt*,PetscInt*,
                  PetscDAPeriodicType*,PetscDAStencilType*)

    int DAGetCorners(PetscDA,
                     PetscInt*,PetscInt*,PetscInt*,
                     PetscInt*,PetscInt*,PetscInt*)
    int DAGetGhostCorners(PetscDA,
                          PetscInt*,PetscInt*,PetscInt*,
                          PetscInt*,PetscInt*,PetscInt*)

    int DASetUniformCoordinates(PetscDA,
                                PetscReal,PetscReal,
                                PetscReal,PetscReal,
                                PetscReal,PetscReal)
    int DASetCoordinates(PetscDA,PetscVec)
    int DAGetCoordinates(PetscDA,PetscVec*)
    int DAGetCoordinateDA(PetscDA,PetscDA*)
    int DAGetGhostedCoordinates(PetscDA,PetscVec*)

    int DACreateGlobalVector(PetscDA,PetscVec*)
    int DACreateNaturalVector(PetscDA,PetscVec*)
    int DACreateLocalVector(PetscDA,PetscVec*)

    int DAGlobalToLocalBegin(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DAGlobalToLocalEnd(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DAGlobalToNaturalBegin(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DAGlobalToNaturalEnd(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DANaturalToGlobalBegin(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DANaturalToGlobalEnd(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToLocalBegin(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToLocalEnd(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToGlobal(PetscDA,PetscVec,PetscInsertMode,PetscVec)
    int DALocalToGlobalBegin(PetscDA,PetscVec,PetscVec)
    int DALocalToGlobalEnd(PetscDA,PetscVec,PetscVec)

    int DAGetMatrix(PetscDA,PetscMatType,PetscMat*)

    int DAGetAO(PetscDA,PetscAO*)
    int DAGetISLocalToGlobalMapping(PetscDA,PetscLGMap*)
    int DAGetISLocalToGlobalMappingBlck(PetscDA,PetscLGMap*)
    int DAGetScatter(PetscDA,PetscScatter*,PetscScatter*,PetscScatter*)

    int DASetRefinementFactor(PetscDA,PetscInt,PetscInt,PetscInt)
    int DAGetRefinementFactor(PetscDA,PetscInt*,PetscInt*,PetscInt*)
    int DARefine(PetscDA,MPI_Comm,PetscDA*)
    int DACoarsen(PetscDA,MPI_Comm,PetscDA*)

    #int DASetFieldName(PetscDA,PetscInt,const_char[])
    #int DAGetFieldName(PetscDA,PetscInt,char**)


cdef inline int DAGetDim(PetscDA da, PetscInt *dim) nogil:
     return DAGetInfo(da, dim,
                      NULL, NULL, NULL,
                      NULL, NULL, NULL,
                      NULL, NULL,
                      NULL, NULL)
