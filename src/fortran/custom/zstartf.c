/*
  This function should be called to be able to use PETSc routines
  from the FORTRAN subroutines, when the main() routine is in C
*/

void PetscInitializeFortran()
{
  int s1,s2,s3;
  s1 = MPIR_FromPointer(STDOUT_VIEWER_SELF);
  s2 = MPIR_FromPointer(STDERR_VIEWER_SELF);
  s3 = MPIR_FromPointer(STDOUT_VIEWER_WORLD);
  petscsetcommonblock_(&s1,&s2,&s3);
}
  
#if defined(__cplusplus)
extern "C" {
#endif

void petscinitializefortran_()
{
  PetscInitializeFortran();
}

#if defined(__cplusplus)
}
#endif
