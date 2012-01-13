import warnings

def get_conf():
    """Parses various PETSc configuration/include files to get data types.

    precision, indices, complexscalars = get_conf()

    Output:
      precision: 'single', 'double', 'longlong' indicates precision of PetscScalar
      indices: '32', '64' indicates bit-size of PetscInt
      complex: True/False indicates whether PetscScalar is complex or not.
    """

    import sys, os
    precision = None
    indices = None
    complexscalars = None

    try:
        petscdir = os.environ['PETSC_DIR']
    except KeyError:
        warnings.warn('Nonexistent or invalid PETSc installation, using defaults')
        return

    try:
        petscdir = os.path.join(petscdir, os.environ['PETSC_ARCH'])
    except KeyError:
        pass

    petscvariables = os.path.join(petscdir, 'conf', 'petscvariables')
    petscconfinclude = os.path.join(petscdir, 'include', 'petscconf.h')

    try:
        fid = file(petscvariables, 'r')
    except IOError:
        warnings.warn('Nonexistent or invalid PETSc installation, using defaults')
        return None, None, None
    else:
        for line in fid:
            if line.startswith('PETSC_PRECISION'):
                precision = line.strip().split('=')[1].strip('\n').strip()

        fid.close()

    try:
        fid = file(petscconfinclude, 'r')
    except IOError:
        warnings.warn('Nonexistent or invalid PETSc installation, using defaults')
        return None, None, None
    else:
        for line in fid:
            if line.startswith('#define PETSC_USE_64BIT_INDICES 1'):
                indices = '64bit'
            elif line.startswith('#define PETSC_USE_COMPLEX 1'):
                complexscalars = True

        if indices is None:
            indices = '32bit'
        if complexscalars is None:
            complexscalars = False
        fid.close()

    return precision, indices, complexscalars
