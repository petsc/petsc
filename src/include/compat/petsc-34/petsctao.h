#define PETSC_INFINITY   PETSC_MAX_REAL/4.0
#define PETSC_NINFINITY -PETSC_INFINITY

typedef PetscObject Tao;

static PetscClassId TAO_CLASSID = 0;
#define TaoInitializePackage() PetscClassIdRegister("TAO",&TAO_CLASSID)
