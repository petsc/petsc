#ifndef PETSC4PY_COMPAT_VIENNACL_H
#define PETSC4PY_COMPAT_VIENNACL_H

#if !defined(PETSC_HAVE_VIENNACL)

#define PetscViennaCLError do { \
    PetscFunctionBegin; \
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires ViennaCL",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode  VecCreateSeqViennaCLWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,PetscScalar cpuarray[],PetscScalar* viennaclvec,Vec *V) {PetscViennaCLError;}
PetscErrorCode VecCreateMPIViennaCLWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,PetscScalar cpuarray[],PetscScalar *viennaclvec,Vec *vv) {PetscViennaCLError;}

#undef PetscViennaCLError

# else

PetscErrorCode VecCreateSeqViennaCLWithArrays(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscScalar[],Vec*);
PetscErrorCode VecCreateMPIViennaCLWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscScalar[],Vec*);

#endif

#endif/*PETSC4PY_COMPAT_VIENNACL_H*/
