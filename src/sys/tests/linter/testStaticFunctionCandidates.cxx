#include "testheader.h"

void BareFunctionShouldGetStatic(void) { }

extern void ExternFunctionShouldNotGetStatic(void) { }

static void StaticFunctionShouldNotGetStatic(void) { }

// this should not get static
static void StaticFunctionPreDeclShouldNotGetStatic(void);

// this should get static!
void StaticFunctionPreDeclShouldNotGetStatic(void) { }

extern void ExternFunctionPreDeclShouldNotGetStatic(void);

void ExternFunctionPreDeclShouldNotGetStatic(void) { }

void BareFunctionPreDeclShouldGetStatic(void);

void BareFunctionPreDeclShouldGetStatic(void) { }

// declaration in testheader has "extern"
void ExternHeaderFunctionShouldNotGetStatic(void) { }

class Foo {
public:
  friend void swap();
};

void swap() { }

// clang-format off
void                                  ExternHeaderBadFormattingShouldNotGetStatic              ( void)
{

}
// clang-format on

static char *StaticPointerShouldNotGetStatic()
{
  return nullptr;
}

char *BarePointerShouldGetStatic()
{
  return nullptr;
}

extern char *ExternPointerShouldGetStatic()
{
  return nullptr;
}

PETSC_EXTERN char *PetscExternPointerShouldNotGetStatic()
{
  return nullptr;
}

PETSC_INTERN char *PetscInternPointerShouldNotGetStatic()
{
  return nullptr;
}

// clang-format off
PETSC_EXTERN char *                   PetscExternPointerBadFormattingShouldNotGetStatic   (   )
{
  return nullptr;
}

PETSC_INTERN char *               PetscInternBadFormattingPointerShouldNotGetStatic ()
{
  return nullptr;
}
// clang-format on

char *PetscExternHeaderPointerShouldNotGetStatic()
{
  return nullptr;
}

char *PetscInternHeaderPointerShouldNotGetStatic()
{
  return nullptr;
}

char *PetscExternHeaderPointerBadFormattingShouldNotGetStatic()
{
  return nullptr;
}

char *PetscInternHeaderPointerBadFormattingShouldNotGetStatic()
{
  return nullptr;
}

// ironically enough, this will get static
void silence_warnings(void)
{
  (void)StaticFunctionShouldNotGetStatic;
  (void)StaticFunctionPreDeclShouldNotGetStatic;
  (void)StaticPointerShouldNotGetStatic;
}
