#ifdef PETSC_HAVE_ICL

__declspec(cpu_specific(pentium_iii))
int PetscSSEIsEnabled(void) {
  return(1);
}

__declspec(cpu_specific(generic))
int PetscSSEIsEnabled(void) {
  return(0);
}

__declspec(cpu_dispatch(generic,pentium_iii))
int PetscSSEIsEnabled(void) {}

#else

int PetscSSEIsEnabled(void) {
  return(0);
}

#endif
