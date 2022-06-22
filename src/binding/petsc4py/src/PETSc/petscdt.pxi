# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef enum int "PetscGaussLobattoLegendreCreateType":
        PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA
        PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON

    int PetscFECreateDefault(MPI_Comm, PetscInt, PetscInt, PetscBool, const char [], PetscInt, PetscFE*)
    int PetscQuadratureCreate(MPI_Comm, PetscQuadrature*)
    int PetscQuadratureDuplicate(PetscQuadrature, PetscQuadrature*)
    int PetscQuadratureGetOrder(PetscQuadrature, PetscInt*)
    int PetscQuadratureSetOrder(PetscQuadrature, PetscInt)
    int PetscQuadratureGetNumComponents(PetscQuadrature, PetscInt*)
    int PetscQuadratureSetNumComponents(PetscQuadrature, PetscInt)
    int PetscQuadratureGetData(PetscQuadrature, PetscInt*, PetscInt*, PetscInt*, const PetscReal *[], const PetscReal *[])
    int PetscQuadratureSetData(PetscQuadrature, PetscInt, PetscInt, PetscInt, const PetscReal [], const PetscReal [])


    int PetscQuadratureView(PetscQuadrature, PetscViewer)
    int PetscQuadratureDestroy(PetscQuadrature *)

    int PetscQuadratureExpandComposite(PetscQuadrature, PetscInt, const PetscReal[], const PetscReal[], PetscQuadrature *)

    int PetscDTLegendreEval(PetscInt,const PetscReal*,PetscInt,const PetscInt*,PetscReal*,PetscReal*,PetscReal*)
    int PetscDTGaussQuadrature(PetscInt,PetscReal,PetscReal,PetscReal*,PetscReal*)
    int PetscDTGaussLobattoLegendreQuadrature(PetscInt,PetscGaussLobattoLegendreCreateType,PetscReal*,PetscReal*)
    int PetscDTReconstructPoly(PetscInt,PetscInt,const PetscReal*,PetscInt,const PetscReal*,PetscReal*)
    int PetscDTGaussTensorQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*)
    int PetscDTGaussJacobiQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*)

    int PetscDTTanhSinhTensorQuadrature(PetscInt, PetscInt, PetscReal, PetscReal, PetscQuadrature *)
    int PetscDTTanhSinhIntegrate(void (*)(PetscReal *, void *, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *)
    int PetscDTTanhSinhIntegrateMPFR(void (*)(PetscReal *, void *, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *)

    int PetscGaussLobattoLegendreIntegrate(PetscInt, PetscReal *, PetscReal *, const PetscReal *, PetscReal *)
    int PetscGaussLobattoLegendreElementLaplacianCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    int PetscGaussLobattoLegendreElementLaplacianDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    int PetscGaussLobattoLegendreElementGradientCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***)
    int PetscGaussLobattoLegendreElementGradientDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***)
    int PetscGaussLobattoLegendreElementAdvectionCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    int PetscGaussLobattoLegendreElementAdvectionDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    int PetscGaussLobattoLegendreElementMassCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
    int PetscGaussLobattoLegendreElementMassDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***)
