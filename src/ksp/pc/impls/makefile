
ALL: lib

LIBBASE  = libpetscksp
DIRS     = jacobi none sor shell bjacobi mg eisens asm ksp composite redundant spai is pbjacobi ml\
           mat hypre tfs fieldsplit factor galerkin cp wb python ainvcusp sacusp bicgstabcusp \
           chowiluviennacl chowiluviennaclcuda rowscalingviennacl rowscalingviennaclcuda saviennacl saviennaclcuda\
           lsc redistribute gasm svd gamg parms bddc kaczmarz telescope
LOCDIR   = src/ksp/pc/impls/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

