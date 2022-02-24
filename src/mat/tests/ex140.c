static char help[] = "Tests MATPYTHON from C\n\n";

#include <petscmat.h>
/* MATPYTHON has support for wrapping these operations
   MatHasOperation_Python inspects the user's Python class and checks
   if the methods are provided */
MatOperation optenum[] = {MATOP_MULT,
                          MATOP_MULT_ADD,
                          MATOP_MULT_TRANSPOSE,
                          MATOP_MULT_TRANSPOSE_ADD,
                          MATOP_SOLVE,
                          MATOP_SOLVE_ADD,
                          MATOP_SOLVE_TRANSPOSE,
                          MATOP_SOLVE_TRANSPOSE_ADD,
                          MATOP_SOR,
                          MATOP_GET_DIAGONAL,
                          MATOP_DIAGONAL_SCALE,
                          MATOP_NORM,
                          MATOP_ZERO_ENTRIES,
                          MATOP_GET_DIAGONAL_BLOCK,
                          MATOP_DUPLICATE,
                          MATOP_COPY,
                          MATOP_SCALE,
                          MATOP_SHIFT,
                          MATOP_DIAGONAL_SET,
                          MATOP_ZERO_ROWS_COLUMNS,
                          MATOP_CREATE_SUBMATRIX,
                          MATOP_CREATE_VECS,
                          MATOP_CONJUGATE,
                          MATOP_REAL_PART,
                          MATOP_IMAGINARY_PART,
                          MATOP_MISSING_DIAGONAL,
                          MATOP_MULT_DIAGONAL_BLOCK,
                          MATOP_MULT_HERMITIAN_TRANSPOSE,
                          MATOP_MULT_HERMITIAN_TRANS_ADD};

/* Name of the methods in the user's Python class */
const char* const optstr[] = {"mult",
                              "multAdd",
                              "multTranspose",
                              "multTransposeAdd",
                              "solve",
                              "solveAdd",
                              "solveTranspose",
                              "solveTransposeAdd",
                              "SOR",
                              "getDiagonal",
                              "diagonalScale",
                              "norm",
                              "zeroEntries",
                              "getDiagonalBlock",
                              "duplicate",
                              "copy",
                              "scale",
                              "shift",
                              "setDiagonal",
                              "zeroRowsColumns",
                              "createSubMatrix",
                              "getVecs",
                              "conjugate",
                              "realPart",
                              "imagPart",
                              "missingDiagonal",
                              "multDiagonalBlock",
                              "multHermitian",
                              "multHermitianAdd"};

PetscErrorCode RunHasOperationTest()
{
  Mat A;
  PetscInt matop, nop = sizeof(optenum)/sizeof(PetscInt);

  PetscFunctionBegin;
  for (matop = 0; matop < nop; matop++) {
    char opts[256];
    PetscBool hasop;
    PetscInt i;

    CHKERRQ(PetscSNPrintf(opts,256,"-enable %s",optstr[matop]));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Testing with %s\n",opts));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
    CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,0,0));
    CHKERRQ(MatSetType(A,MATPYTHON));
    CHKERRQ(MatPythonSetType(A,"ex140.py:Matrix"));
    /* default case, no user implementation */
    for (i = 0; i < nop; i++) {
      CHKERRQ(MatHasOperation(A,optenum[i],&hasop));
      if (hasop) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Error: %s present\n",optstr[i]));
      } else {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Pass: %s\n",optstr[i]));
      }
    }
    /* customize Matrix class at a later stage and add support for optenum[matop] */
    CHKERRQ(PetscOptionsInsertString(NULL,opts));
    CHKERRQ(MatSetFromOptions(A));
    for (i = 0; i < nop; i++) {
      CHKERRQ(MatHasOperation(A,optenum[i],&hasop));
      if (hasop && i != matop) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Error: %s present\n",optstr[i]));
      } else if (!hasop && i == matop) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Error: %s not present\n",optstr[i]));
      } else {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Pass: %s\n",optstr[i]));
      }
    }
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(PetscOptionsClearValue(NULL,opts));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  CHKERRQ(PetscPythonInitialize(NULL,NULL));
  CHKERRQ(RunHasOperationTest();PetscPythonPrintError());
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: petsc4py
      localrunfiles: ex140.py

TEST*/
