PetscErrorCode MatGetRowIJ_Compat(Mat mat,PetscInt shift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done) { return MatGetRowIJ(mat,shift,symmetric,inodecompressed,n,(PetscInt**)ia,(PetscInt**)ja,done);}
#define MatGetRowIJ MatGetRowIJ_Compat
PetscErrorCode MatRestoreRowIJ_Compat(Mat mat,PetscInt shift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done) { return MatRestoreRowIJ(mat,shift,symmetric,inodecompressed,n,(PetscInt**)ia,(PetscInt**)ja,done);}
#define MatRestoreRowIJ MatRestoreRowIJ_Compat

PetscErrorCode MatGetColumnIJ_Compat(Mat mat,PetscInt shift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done) { return MatGetColumnIJ(mat,shift,symmetric,inodecompressed,n,(PetscInt**)ia,(PetscInt**)ja,done);}
#define MatGetColumnIJ MatGetColumnIJ_Compat
PetscErrorCode MatRestoreColumnIJ_Compat(Mat mat,PetscInt shift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done) { return MatRestoreColumnIJ(mat,shift,symmetric,inodecompressed,n,(PetscInt**)ia,(PetscInt**)ja,done);}
#define MatRestoreColumnIJ MatRestoreColumnIJ_Compat
