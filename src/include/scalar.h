static PyObject *PyPetscScalar_FromPetscScalar(PetscScalar s)
{
#if defined(PETSC_USE_COMPLEX)
  double a = (double) PetscRealPart(s);
  double b = (double) PetscImaginaryPart(s);
  return PyComplex_FromDoubles(a, b);
#else
  return PyFloat_FromDouble((double)s);
#endif
}

static PetscScalar PyPetscScalar_AsPetscScalar(PyObject *o)
{
#if defined(PETSC_USE_COMPLEX)
  Py_complex cval = PyComplex_AsCComplex(o);
  PetscReal a = (PetscReal) cval.real;
  PetscReal b = (PetscReal) cval.imag;
  return a + b * PETSC_i;
#else
  return PyFloat_AsDouble(o);
#endif
}
