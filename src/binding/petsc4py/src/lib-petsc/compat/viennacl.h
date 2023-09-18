#ifndef PETSC4PY_COMPAT_VIENNACL_H
#define PETSC4PY_COMPAT_VIENNACL_H

#if !defined(PETSC_HAVE_VIENNACL)

#define PetscViennaCLError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires ViennaCL",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode VecCreateSeqViennaCLWithArrays(PETSC_UNUSED MPI_Comm comm,PETSC_UNUSED PetscInt bs,PETSC_UNUSED PetscInt n,PETSC_UNUSED PetscScalar cpuarray[],PETSC_UNUSED PetscScalar* viennaclvec,PETSC_UNUSED Vec *V) {PetscViennaCLError;}
PetscErrorCode VecCreateMPIViennaCLWithArrays(PETSC_UNUSED MPI_Comm comm,PETSC_UNUSED PetscInt bs,PETSC_UNUSED PetscInt n,PETSC_UNUSED PetscInt N,PETSC_UNUSED PetscScalar cpuarray[],PETSC_UNUSED PetscScalar *viennaclvec,PETSC_UNUSED Vec *vv) {PetscViennaCLError;}

#undef PetscViennaCLError

# else

PetscErrorCode VecCreateSeqViennaCLWithArrays(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscScalar[],Vec*);
PetscErrorCode VecCreateMPIViennaCLWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscScalar[],Vec*);

#endif

#endif/*PETSC4PY_COMPAT_VIENNACL_H*/
