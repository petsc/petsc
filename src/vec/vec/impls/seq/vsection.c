/*
   This file contains routines for basic section object implementation.
*/

#include <private/vecimpl.h>   /*I  "petscvec.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "PetscSectionCreate"
/*@C
  PetscSectionCreate - Allocates PetscSection space and sets the map contents to the default.

  Collective on MPI_Comm

  Input Parameters:
+ comm - the MPI communicator
- s    - pointer to the section

  Level: developer

  Notes: Typical calling sequence
       PetscLayoutCreate(MPI_Comm,PetscLayout *);
       PetscLayoutSetBlockSize(PetscLayout,1);
       PetscLayoutSetSize(PetscLayout,n) or PetscLayoutSetLocalSize(PetscLayout,N);
       PetscLayoutSetUp(PetscLayout);
       PetscLayoutGetSize(PetscLayout,PetscInt *); or PetscLayoutGetLocalSize(PetscLayout,PetscInt *;)
       PetscLayoutDestroy(PetscLayout);

       The PetscSection object and methods are intended to be used in the PETSc Vec and Mat implementions; it is
       recommended they not be used in user codes unless you really gain something in their use.

  Fortran Notes:
      Not available from Fortran

.seealso: PetscSection, PetscSectionDestroy()
@*/
PetscErrorCode PetscSectionCreate(MPI_Comm comm, PetscSection *s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscSection, s);CHKERRQ(ierr);
  (*s)->atlasLayout.comm   = comm;
  (*s)->atlasLayout.pStart = -1;
  (*s)->atlasLayout.pEnd   = -1;
  (*s)->atlasLayout.numDof = 1;
  (*s)->atlasDof           = PETSC_NULL;
  (*s)->atlasOff           = PETSC_NULL;
  (*s)->bc                 = PETSC_NULL;
  (*s)->bcIndices          = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionCheckConstraints"
PetscErrorCode PetscSectionCheckConstraints(PetscSection s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!s->bc) {
    ierr = PetscSectionCreate(s->atlasLayout.comm, &s->bc);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(s->bc, s->atlasLayout.pStart, s->atlasLayout.pEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetChart"
PetscErrorCode PetscSectionGetChart(PetscSection s, PetscInt *pStart, PetscInt *pEnd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pStart) {*pStart = s->atlasLayout.pStart;}
  if (pEnd)   {*pEnd   = s->atlasLayout.pEnd;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetChart"
PetscErrorCode PetscSectionSetChart(PetscSection s, PetscInt pStart, PetscInt pEnd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s->atlasLayout.pStart = pStart;
  s->atlasLayout.pEnd   = pEnd;
  ierr = PetscFree2(s->atlasDof, s->atlasOff);CHKERRQ(ierr);
  ierr = PetscMalloc2((pEnd - pStart), PetscInt, &s->atlasDof, (pEnd - pStart), PetscInt, &s->atlasOff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetDof"
PetscErrorCode PetscSectionGetDof(PetscSection s, PetscInt point, PetscInt *numDof)
{
  PetscFunctionBegin;
  if ((point < s->atlasLayout.pStart) || (point >= s->atlasLayout.pEnd)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %p should be in [%d, %d)", point, s->atlasLayout.pStart, s->atlasLayout.pEnd);
  }
  *numDof = s->atlasDof[point - s->atlasLayout.pStart];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetDof"
PetscErrorCode PetscSectionSetDof(PetscSection s, PetscInt point, PetscInt numDof)
{
  PetscFunctionBegin;
  if ((point < s->atlasLayout.pStart) || (point >= s->atlasLayout.pEnd)) {
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Section point %p should be in [%d, %d)", point, s->atlasLayout.pStart, s->atlasLayout.pEnd);
  }
  s->atlasDof[point - s->atlasLayout.pStart] = numDof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetConstraintDof"
PetscErrorCode PetscSectionGetConstraintDof(PetscSection s, PetscInt point, PetscInt *numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->bc) {
    ierr = PetscSectionGetDof(s->bc, point, numDof);CHKERRQ(ierr);
  } else {
    *numDof = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetConstraintDof"
PetscErrorCode PetscSectionSetConstraintDof(PetscSection s, PetscInt point, PetscInt numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (numDof) {
    ierr = PetscSectionCheckConstraints(s);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(s->bc, point, numDof);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetUp"
PetscErrorCode PetscSectionSetUp(PetscSection s)
{
  PetscInt       offset = 0, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(p = 0; p < s->atlasLayout.pEnd - s->atlasLayout.pStart; ++p) {
    s->atlasOff[p] = offset;
    offset += s->atlasDof[p];
  }
  if (s->bc) {
    const PetscInt last = (s->bc->atlasLayout.pEnd-s->bc->atlasLayout.pStart) - 1;

    ierr = PetscSectionSetUp(s->bc);CHKERRQ(ierr);
    ierr = PetscMalloc((s->bc->atlasOff[last] + s->bc->atlasDof[last]) * sizeof(PetscInt), &s->bcIndices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionDestroy - Frees a section object and frees its range if that exists.

  Collective on MPI_Comm

  Input Parameters:
. s - the PetscSection

  Level: developer

    The PetscSection object and methods are intended to be used in the PETSc Vec and Mat implementions; it is
    recommended they not be used in user codes unless you really gain something in their use.

  Fortran Notes:
    Not available from Fortran

.seealso: PetscSection, PetscSectionCreate()
@*/
#undef __FUNCT__
#define __FUNCT__ "PetscSectionDestroy"
PetscErrorCode  PetscSectionDestroy(PetscSection *s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*s) PetscFunctionReturn(0);
  if (!(*s)->refcnt--) {
    ierr = PetscSectionDestroy(&(*s)->bc);CHKERRQ(ierr);
    ierr = PetscFree((*s)->bcIndices);CHKERRQ(ierr);
    ierr = PetscFree2((*s)->atlasDof, (*s)->atlasOff);CHKERRQ(ierr);
    ierr = PetscFree((*s));CHKERRQ(ierr);
  }
  *s = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetValuesSection"
PetscErrorCode VecGetValuesSection(Vec v, PetscSection s, PetscInt point, PetscScalar **values)
{
  PetscScalar   *baseArray;
  const PetscInt p = point - s->atlasLayout.pStart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(v, &baseArray);CHKERRQ(ierr);
  *values = &baseArray[s->atlasOff[p]];
  ierr = VecRestoreArray(v, &baseArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecIntGetValuesSection"
PetscErrorCode VecIntGetValuesSection(PetscInt *baseArray, PetscSection s, PetscInt point, PetscInt **values)
{
  const PetscInt p = point - s->atlasLayout.pStart;

  PetscFunctionBegin;
  *values = &baseArray[s->atlasOff[p]];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetValuesSection"
PetscErrorCode VecSetValuesSection(Vec v, PetscSection s, PetscInt point, PetscScalar values[], InsertMode mode)
{
  PetscScalar    *baseArray, *array;
  const PetscBool doInsert    = mode == INSERT_VALUES     || mode == INSERT_ALL_VALUES ? PETSC_TRUE : PETSC_FALSE;
  const PetscBool doBC        = mode == INSERT_ALL_VALUES || mode == ADD_ALL_VALUES    ? PETSC_TRUE : PETSC_FALSE;
  const PetscInt  p           = point - s->atlasLayout.pStart;
  const PetscInt  orientation = 0; /* Needs to be included for use in closure operations */
  PetscInt        cDim        = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetConstraintDof(s, p, &cDim);CHKERRQ(ierr);
  ierr = VecGetArray(v, &baseArray);CHKERRQ(ierr);
  array = &baseArray[s->atlasOff[p]];
  if (!cDim) {
    if (orientation >= 0) {
      const PetscInt dim = s->atlasDof[p];
      PetscInt       i;

      if (doInsert) {
        for(i = 0; i < dim; ++i) {
          array[i] = values[i];
        }
      } else {
        for(i = 0; i < dim; ++i) {
          array[i] += values[i];
        }
      }
    } else {
#if 0
      int offset = 0;
      int j      = -1;

      for(int space = 0; space < this->getNumSpaces(); ++space) {
        const int& dim = this->getFiberDimension(p, space);

        for(int i = dim-1; i >= 0; --i) {
          array[++j] = values[i+offset];
        }
        offset += dim;
      }
#else
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Fibration is not yet implemented in PetscSection");
#endif
    }
  } else {
    if (orientation >= 0) {
      const PetscInt  dim  = s->atlasDof[p];
      PetscInt        cInd = 0, i;
      PetscInt       *cDof;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      if (doInsert) {
        for(i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {
            if (doBC) {array[i] = values[i];} /* Constrained update */
            ++cInd;
            continue;
          }
          array[i] = values[i]; /* Unconstrained update */
        }
      } else {
        for(i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {
            if (doBC) {array[i] += values[i];} /* Constrained update */
            ++cInd;
            continue;
          }
          array[i] += values[i]; /* Unconstrained update */
        }
      }
    } else {
#if 0
      const PetscInt *cDof;
      PetscInt        offset  = 0;
      PetscInt        cOffset = 0;
      PetscInt        j       = 0, space;

      ierr = PetscSectionGetConstraintDof(s, point, &cDof);CHKERRQ(ierr);
      for(space = 0; space < this->getNumSpaces(); ++space) {
        const PetscInt  dim = this->getFiberDimension(p, space);
        const PetscInt tDim = this->getConstrainedFiberDimension(p, space);
        const PetscInt sDim = dim - tDim;
        PetscInt       cInd = 0, i ,k;

        for(i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
          if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
          array[j] = values[k];
        }
        offset  += dim;
        cOffset += dim - tDim;
      }
#else
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Fibration is not yet implemented in PetscSection");
#endif
    }
  }
  ierr = VecRestoreArray(v, &baseArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecIntSetValuesSection"
PetscErrorCode VecIntSetValuesSection(PetscInt *baseArray, PetscSection s, PetscInt point, PetscInt values[], InsertMode mode)
{
  PetscInt      *array;
  const PetscInt p           = point - s->atlasLayout.pStart;
  const PetscInt orientation = 0; /* Needs to be included for use in closure operations */
  PetscInt       cDim        = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetConstraintDof(s, p, &cDim);CHKERRQ(ierr);
  array = &baseArray[s->atlasOff[p]];
  if (!cDim) {
    if (orientation >= 0) {
      const PetscInt dim = s->atlasDof[p];
      PetscInt       i;

      if (mode == INSERT_VALUES) {
        for(i = 0; i < dim; ++i) {
          array[i] = values[i];
        }
      } else {
        for(i = 0; i < dim; ++i) {
          array[i] += values[i];
        }
      }
    } else {
#if 0
      int offset = 0;
      int j      = -1;

      for(int space = 0; space < this->getNumSpaces(); ++space) {
        const int& dim = this->getFiberDimension(p, space);

        for(int i = dim-1; i >= 0; --i) {
          array[++j] = values[i+offset];
        }
        offset += dim;
      }
#else
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Fibration is not yet implemented in PetscSection");
#endif
    }
  } else {
    if (orientation >= 0) {
      const PetscInt dim  = s->atlasDof[p];
      PetscInt       cInd = 0, i;
      PetscInt      *cDof;

      ierr = PetscSectionGetConstraintIndices(s, point, &cDof);CHKERRQ(ierr);
      if (mode == INSERT_VALUES) {
        for(i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
          array[i] = values[i];
        }
      } else {
        for(i = 0; i < dim; ++i) {
          if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
          array[i] += values[i];
        }
      }
    } else {
#if 0
      const PetscInt *cDof;
      PetscInt        offset  = 0;
      PetscInt        cOffset = 0;
      PetscInt        j       = 0, space;

      ierr = PetscSectionGetConstraintDof(s, point, &cDof);CHKERRQ(ierr);
      for(space = 0; space < this->getNumSpaces(); ++space) {
        const PetscInt  dim = this->getFiberDimension(p, space);
        const PetscInt tDim = this->getConstrainedFiberDimension(p, space);
        const PetscInt sDim = dim - tDim;
        PetscInt       cInd = 0, i ,k;

        for(i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
          if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
          array[j] = values[k];
        }
        offset  += dim;
        cOffset += dim - tDim;
      }
#else
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Fibration is not yet implemented in PetscSection");
#endif
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetConstraintIndices"
PetscErrorCode PetscSectionGetConstraintIndices(PetscSection s, PetscInt point, PetscInt **indices)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->bc) {
    ierr = VecIntGetValuesSection(s->bcIndices, s->bc, point, indices);CHKERRQ(ierr);
  } else {
    *indices = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSectionSetConstraintIndices"
PetscErrorCode PetscSectionSetConstraintIndices(PetscSection s, PetscInt point, PetscInt indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->bc) {
    ierr = VecIntSetValuesSection(s->bcIndices, s->bc, point, indices, INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
