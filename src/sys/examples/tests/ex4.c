#if !defined(PETSC_USE_CXX_COMPLEX_FLOAT_WORKAROUND)
#define PETSC_USE_CXX_COMPLEX_FLOAT_WORKAROUND 1
#endif

#include <petscsys.h>

static char help[] = "Test PetscComplex binary operators.\n";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInitialize(&argc,&argv,NULL,help);
#if defined(PETSC_HAVE_COMPLEX)
  {
    int          i = 2;
    float        f = 2;
    double       d = 2;
    PetscInt     j = 2;
    PetscReal    r = 2;
    PetscComplex z;

#define TestOps(BOP,IOP) do {                                             \
    z = i; z = z BOP i; z = i BOP z; z IOP i; (void)(z==i); (void)(z!=i); \
    z = f; z = z BOP f; z = f BOP z; z IOP f; (void)(z==f); (void)(z!=f); \
    z = d; z = z BOP d; z = d BOP z; z IOP d; (void)(z==d); (void)(z!=d); \
    z = j; z = z BOP j; z = r BOP z; z IOP j; (void)(z==j); (void)(z!=j); \
    z = r; z = z BOP r; z = r BOP z; z IOP r; (void)(z==r); (void)(z!=r); } while (0)

    TestOps(+,+=);
    TestOps(-,-=);
    TestOps(*,*=);
    TestOps(/,/=);
  }
#endif
  PetscFinalize();
  return 0;
}
