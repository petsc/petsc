#pragma once

/* MANSEC = Vec */
/* SUBMANSEC = PetscSection */

/*S
  PetscSection - Provides a mapping from integers in a designated domain (defined by bounds `startp` to `endp`) to integers which can then be used
  for accessing entries in arrays, other `PetscSection`s, `IS`s, `Vec`s, and `Mat`s.

  One can think of `PetscSection` as a library-based tool for indexing into multi-dimensional jagged arrays which is needed
  since programming languages do not provide jagged array functionality baked into their syntax.

  The domain, `startp` to `endp`, is called the chart of the `PetscSection()` and is set with `PetscSectionSetChart()` and accessed
  `PetscSectionGetChart()`. `startp` does not need to  be 0, `endp` must be greater than or equal to `startp` and the bounds
  may be positive or negative.

  The range of a `PetscSection` is in the space of
  contiguous sets of integers. These ranges are frequently interpreted as domains (charts, meaning lower and upper bounds) of other array-like objects,
  especially other `PetscSection`s, `IS`s, and `Vec`s.

  For each point in the chart (from `startp` to `endp`) of a `PetscSection`, the output set is represented through an `offset` and a
  `count`, which can be obtained using `PetscSectionGetOffset()` and `PetscSectionGetDof()` respectively and can be set via
  `PetscSectionSetOffset()` and `PetscSectionSetDof()`. Lookup is typically using
  accessors or routines like `VecGetValuesSection()`

  The indices returned by the `PetscSection`
  are appropriate for the kind of `Vec` it is associated with. For example, if the vector being indexed is a local vector, we call the section a
  local section. If the section indexes a global vector, we call it a global section. For parallel vectors, like global vectors, we use negative
  indices to indicate dofs owned by other processes.

  Typically `PetscSections` are first constructed via a series of calls to `PetscSectionSetOffset()` and `PetscSectionSetDof()`, finalized via
  a call to `PetscSectionSetup()` and then used to index into arrays and other PETSc objects. The construction (setup) phase corresponds to providing all
  the information needed to define the multi-dimensional jagged array structure.

  `PetscSection` is used heavily by `DMPLEX`. Simplier `DM`, such as `DMDA`, generally do not need `PetscSection` since their array access patterns
  are simplier and can be fully expressed using standard programming language array syntax, see [DM commonality](ch_dmcommonality).

  Level: beginner

.seealso: [PetscSection](ch_petscsection), `PetscSectionCreate()`, `PetscSectionGetOffset()`, `PetscSectionGetOffset()`, `PetscSectionSetChart()`,
          `PetscSectionGetChart()`, `PetscSectionDestroy()`, `PetscSectionSym`, `PetscSectionSetup()`, `DM`, `DMDA`, `DMPLEX`
S*/
typedef struct _p_PetscSection *PetscSection;

/*S
  PetscSectionSym - Symmetries of the data referenced by a `PetscSection`.

  Often the order of data index by a `PetscSection` is meaningful, and describes additional structure, such as points on a
  line, grid, or lattice.  If the data is accessed from a different "orientation", then the image of the data under
  access then undergoes a symmetry transformation.  A `PetscSectionSym` specifies these symmetries.  The types of
  symmetries that can be specified are of the form R * P, where R is a diagonal matrix of scalars, and P is a permutation.

  Level: developer

.seealso: [PetscSection](ch_petscsection), `PetscSection`, `PetscSectionSymCreate()`, `PetscSectionSymDestroy()`, `PetscSectionSetSym()`, `PetscSectionGetSym()`, `PetscSectionSetFieldSym()`,
          `PetscSectionGetFieldSym()`, `PetscSectionGetSymPoints()`, `PetscSectionSymType`, `PetscSectionSymSetType()`, `PetscSectionSymGetType()`
S*/
typedef struct _p_PetscSectionSym *PetscSectionSym;

/*J
  PetscSectionSymType - String with the name of a `PetscSectionSym` type.

  Level: developer

  Note:
  `PetscSectionSym` has no default implementation, but is used by `DM` in `PetscSectionSymCreateLabel()`.

.seealso: [PetscSection](ch_petscsection), `PetscSectionSymSetType()`, `PetscSectionSymGetType()`, `PetscSectionSym`, `PetscSectionSymCreate()`, `PetscSectionSymRegister()`
J*/
typedef const char *PetscSectionSymType;
