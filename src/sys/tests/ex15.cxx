static char help[] = "Tests inclusion of petscsystypes.h.\n\n";

#include <petscsys.h>

#if defined(PETSC_HAVE_COMPLEX)
template <class Type>
PetscErrorCode TestComplexOperators(Type x, PetscBool check, double &ans)
{
  double       res;
  PetscComplex z = x;

  PetscFunctionBeginUser;
  (void)z;
  z = x;
  z += x;
  z = z + x;
  z = x + z;
  z = x;
  z -= x;
  z = z - x;
  z = x - z;
  z = x;
  z *= x;
  z = z * x;
  z = x * z;
  z = x;
  z /= x;
  z = z / x;
  z = x / z;
  (void)(z == x);
  (void)(x == z);
  (void)(z != x);
  (void)(x != z);
  res = PetscRealPartComplex(z);
  if (check) PetscCheck(PetscAbs(ans - res) < 1e-5, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected %g, but get incorrect result %g", ans, res);
  else ans = res;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

int main(int argc, char **argv)
{
  /* numeric types */
  PetscScalar svalue;
  PetscReal   rvalue;
#if defined(PETSC_HAVE_COMPLEX)
  PetscComplex cvalue;
#endif

  /* integer types */
  PetscInt64   i64;
  PetscInt     i;
  PetscBLASInt bi;
  PetscMPIInt  rank;

  /* PETSc types */
  PetscBool        b;
  PetscErrorCode   ierr;
  PetscClassId     cid;
  PetscEnum        e;
  PetscShort       s;
  char             c;
  PetscFloat       f;
  PetscLogDouble   ld;
  PetscObjectId    oid;
  PetscObjectState ost;

  /* Enums */
  PetscCopyMode          cp;
  PetscDataType          dt;
  PetscFileMode          fm;
  PetscDLMode            dlm;
  PetscBinarySeekType    bsk;
  PetscBuildTwoSidedType b2s;
  InsertMode             im;
  PetscSubcommType       subct;

  /* Sys objects */
  PetscObject             obj;
  PetscRandom             rand;
  PetscToken              token;
  PetscFunctionList       flist;
  PetscDLHandle           dlh;
  PetscObjectList         olist;
  PetscDLLibrary          dlist;
  PetscContainer          cont;
  PetscSubcomm            subc;
  PetscHeap               pheap;
  PetscShmComm            scomm;
  PetscOmpCtrl            octrl;
  PetscSegBuffer          sbuff;
  PetscOptionsHelpPrinted oh;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  svalue = 0.0;
  rvalue = 0.0;
#if defined(PETSC_HAVE_COMPLEX)
  cvalue = 0.0;
#endif

#if defined(PETSC_HAVE_COMPLEX)
  double ans = 0.0;

  // PetscComplex .op. integer
  PetscCall(TestComplexOperators((PetscReal)1.0, PETSC_FALSE, ans)); // assuming with PetscReal, we get a correct answer
  PetscCall(TestComplexOperators((char)1, PETSC_TRUE, ans));         // check against the answer
  PetscCall(TestComplexOperators((signed char)1, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((signed short)1, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((signed int)1, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((signed long)1, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((signed long long)1, PETSC_TRUE, ans));

  PetscCall(TestComplexOperators((unsigned char)1, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((unsigned short)1, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((unsigned int)1, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((unsigned long)1, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((unsigned long long)1, PETSC_TRUE, ans));

  // PetscComplex .op. floating point
  PetscCall(TestComplexOperators((PetscReal)0.5, PETSC_FALSE, ans)); // get an answer again
  #if defined(PETSC_HAVE_REAL___FP16)
  PetscCall(TestComplexOperators((__fp16)0.5, PETSC_TRUE, ans));
  #endif
  PetscCall(TestComplexOperators((float)0.5, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((double)0.5, PETSC_TRUE, ans));
  PetscCall(TestComplexOperators((long double)0.5, PETSC_TRUE, ans));
  #if defined(PETSC_HAVE_REAL___FLOAT128)
  PetscCall(TestComplexOperators((__float128)0.5, PETSC_TRUE, ans));
  #endif

#endif

  i64  = 0;
  i    = 0;
  bi   = 0;
  rank = 0;

  b   = PETSC_FALSE;
  cid = 0;
  e   = ENUM_DUMMY;
  s   = 0;
  c   = '\0';
  f   = 0;
  ld  = 0.0;
  oid = 0;
  ost = 0;

  cp    = PETSC_COPY_VALUES;
  dt    = PETSC_DATATYPE_UNKNOWN;
  fm    = FILE_MODE_READ;
  dlm   = PETSC_DL_DECIDE;
  bsk   = PETSC_BINARY_SEEK_SET;
  b2s   = PETSC_BUILDTWOSIDED_NOTSET;
  im    = INSERT_VALUES;
  subct = PETSC_SUBCOMM_GENERAL;

  obj   = nullptr;
  rand  = nullptr;
  token = nullptr;
  flist = nullptr;
  dlh   = nullptr;
  olist = nullptr;
  dlist = nullptr;
  cont  = nullptr;
  subc  = nullptr;
  pheap = nullptr;
  scomm = nullptr;
  octrl = nullptr;
  sbuff = nullptr;
  oh    = nullptr;

  /* prevent to issue warning about unused-but-set variables */
  (void)help;

  (void)svalue;
  (void)rvalue;
#if defined(PETSC_HAVE_COMPLEX)
  (void)cvalue;
#endif
  (void)i64;
  (void)i;
  (void)bi;
  (void)rank;

  (void)b;
  (void)ierr;
  (void)cid;
  (void)e;
  (void)s;
  (void)c;
  (void)f;
  (void)ld;
  (void)oid;
  (void)ost;

  (void)cp;
  (void)dt;
  (void)fm;
  (void)dlm;
  (void)bsk;
  (void)b2s;
  (void)im;
  (void)subct;

  (void)obj;
  (void)rand;
  (void)token;
  (void)flist;
  (void)dlh;
  (void)olist;
  (void)dlist;
  (void)cont;
  (void)subc;
  (void)pheap;
  (void)scomm;
  (void)octrl;
  (void)sbuff;
  (void)oh;
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    output_file: output/empty.out

TEST*/
