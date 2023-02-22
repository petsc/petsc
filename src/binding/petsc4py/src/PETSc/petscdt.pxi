# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef enum int "PetscGaussLobattoLegendreCreateType":
        PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA
        PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON

    PetscErrorCode PetscFECreateDefault(MPI_Comm, PetscInt, PetscInt, PetscBool, const char [], PetscInt, PetscFE*)
    PetscErrorCode PetscQuadratureCreate(MPI_Comm, PetscQuadrature*)
    PetscErrorCode PetscQuadratureDuplicate(PetscQuadrature, PetscQuadrature*)
    PetscErrorCode PetscQuadratureGetOrder(PetscQuadrature, PetscInt*)
    PetscErrorCode PetscQuadratureSetOrder(PetscQuadrature, PetscInt)
    PetscErrorCode PetscQuadratureGetNumComponents(PetscQuadrature, PetscInt*)
    PetscErrorCode PetscQuadratureSetNumComponents(PetscQuadrature, PetscInt)
    PetscErrorCode PetscQuadratureGetData(PetscQuadrature, PetscInt*, PetscInt*, PetscInt*, const PetscReal *[], const PetscReal *[])
    PetscErrorCode PetscQuadratureSetData(PetscQuadrature, PetscInt, PetscInt, PetscInt, const PetscReal [], const PetscReal [])


    PetscErrorCode PetscQuadratureView(PetscQuadrature, PetscViewer)
    PetscErrorCode PetscQuadratureDestroy(PetscQuadrature *)

    PetscErrorCode PetscQuadratureExpandComposite(PetscQuadrature, PetscInt, const PetscReal[], const PetscReal[], PetscQuadrature *)

    PetscErrorCode PetscDTLegendreEval(PetscInt,const PetscReal*,PetscInt,const PetscInt*,PetscReal*,PetscReal*,PetscReal*)
    PetscErrorCode PetscDTGaussQuadrature(PetscInt,PetscReal,PetscReal,PetscReal*,PetscReal*)
    PetscErrorCode PetscDTGaussLobattoLegendreQuadrature(PetscInt,PetscGaussLobattoLegendreCreateType,PetscReal*,PetscReal*)
    PetscErrorCode PetscDTReconstructPoly(PetscInt,PetscInt,const PetscReal*,PetscInt,const PetscReal*,PetscReal*)
    PetscErrorCode PetscDTGaussTensorQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*)
    PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*)

    PetscErrorCode PetscDTTanhSinhTensorQuadrature(PetscInt, PetscInt, PetscReal, PetscReal, PetscQuadrature *)
    PetscErrorCode PetscDTTanhSinhIntegrate(void (*)(PetscReal *, void *, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *)
    PetscErrorCode PetscDTTanhSinhIntegrateMPFR(void (*)(PetscReal *, void *, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *)

    PetscErrorCode PetscGaussLobattoLegendreIntegrate(PetscInt, PetscReal *, PetscReal *, const PetscReal *, PetscReal *)
    PetscErrorCode PetscGaussLobattoLegendreElementLaplacianCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    PetscErrorCode PetscGaussLobattoLegendreElementLaplacianDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    PetscErrorCode PetscGaussLobattoLegendreElementGradientCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***)
    PetscErrorCode PetscGaussLobattoLegendreElementGradientDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***)
    PetscErrorCode PetscGaussLobattoLegendreElementAdvectionCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    PetscErrorCode PetscGaussLobattoLegendreElementAdvectionDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    PetscErrorCode PetscGaussLobattoLegendreElementMassCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    PetscErrorCode PetscGaussLobattoLegendreElementMassDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
