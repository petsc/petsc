typedef struct {
  PetscInt    nshift;
  PetscInt    nshift_max;
  PetscReal   shift_lo;
  PetscReal   shift_hi;
  PetscInt    lushift;
  PetscReal   shift_amount;
  PetscReal   shift_top;
  PetscReal   rs;
  PetscScalar pv;
} Shift_Ctx;

EXTERN PetscErrorCode PCLUFactorCheckShift(Mat,MatFactorInfo *,Mat *,Shift_Ctx *,PetscInt *);
