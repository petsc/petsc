const char help[] = "Test PetscAssertPointer type generics";

#include <petsc/private/petscimpl.h>

#define PETSC_TEST_ASSERT_POINTER_GENERICS_SINGLE(type, PETSC_TYPE, string) \
  do { \
    type         *h = NULL; \
    PetscBool     same_string; \
    PetscDataType data_type         = PetscAssertPointer_PetscDataType(h); \
    const char    expected_string[] = string; \
    PetscCall(PetscInfo(NULL, "PetscAssertPointer_PetscDataType(%s *h) = PETSC_%s\n", expected_string, PetscDataTypes[data_type])); \
    PetscCall(PetscInfo(NULL, "PetscAssertPointer_String(%s *h) = \"%s\"\n", expected_string, PetscAssertPointer_String(h))); \
    PetscCheck(data_type == PETSC_TYPE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "[PetscAssertPointer_PetscDataType(%s *h) = %s] != PETSC_%s", expected_string, PetscDataTypes[data_type], PetscDataTypes[PETSC_TYPE]); \
    PetscCall(PetscStrcmp(PetscAssertPointer_String(h), expected_string, &same_string)); \
    PetscCheck(same_string, PETSC_COMM_SELF, PETSC_ERR_PLIB, "[PetscAssertPointer_String(%s *h) = \"%s\"] != \"%s\"", expected_string, PetscAssertPointer_String(h), expected_string); \
  } while (0)
#define PETSC_TEST_ASSERT_POINTER_GENERICS(type, PETSC_TYPE) \
  PETSC_TEST_ASSERT_POINTER_GENERICS_SINGLE(type, PETSC_TYPE, PetscStringize(type)); \
  PETSC_TEST_ASSERT_POINTER_GENERICS_SINGLE(const type, PETSC_TYPE, PetscStringize(type))

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
#if defined(PetscAssertPointer_PetscDataType) && (defined(__cplusplus) || (PETSC_C_VERSION >= 11))
  // clang-format off
  PETSC_TEST_ASSERT_POINTER_GENERICS(          char, PETSC_CHAR   );
  PETSC_TEST_ASSERT_POINTER_GENERICS(   signed char, PETSC_CHAR   );
  PETSC_TEST_ASSERT_POINTER_GENERICS( unsigned char, PETSC_CHAR   );
  PETSC_TEST_ASSERT_POINTER_GENERICS(         short, PETSC_SHORT  );
  PETSC_TEST_ASSERT_POINTER_GENERICS(unsigned short, PETSC_SHORT  );
  PETSC_TEST_ASSERT_POINTER_GENERICS(         float, PETSC_FLOAT  );
  PETSC_TEST_ASSERT_POINTER_GENERICS(        double, PETSC_DOUBLE );
  PETSC_TEST_ASSERT_POINTER_GENERICS(  PetscComplex, PETSC_COMPLEX);
  PETSC_TEST_ASSERT_POINTER_GENERICS(       int32_t, PETSC_INT32  );
  PETSC_TEST_ASSERT_POINTER_GENERICS(      uint32_t, PETSC_INT32  );
  PETSC_TEST_ASSERT_POINTER_GENERICS(       int64_t, PETSC_INT64  );
  PETSC_TEST_ASSERT_POINTER_GENERICS(      uint64_t, PETSC_INT64  );
  // clang-format on
#endif
#if defined(PetscAssertPointer_PetscDataType) && defined(__cplusplus)
  PETSC_TEST_ASSERT_POINTER_GENERICS(PetscBool, PETSC_BOOL);
#endif
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
