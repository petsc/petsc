(ch_petscsection)=

# PetscSection: Connecting Grids to Data

The strongest links between solvers and discretizations are

- the relationship between the layout of data over a mesh (or similar structure) and the data layout in arrays and `Vec` used for computation,
- data partitioning, and
- ordering of data.

To enable modularity, we encode the operations above in simple data
structures that can be understood by the linear algebraic and solver components of PETSc (`Vec`, `Mat`, `KSP`, `PC`, `SNES`, `TS`, `Tao`, `PetscRegressor`)
without explicit reference to the mesh (topology) or discretization (analysis).

While `PetscSection` is currently only employed for `DMPlex`, `DMForest`, and `DMNetwork` mesh descriptions, much of its operation is general enough to be utilized for other types of discretizations.
This section will explain the basic concepts of a `PetscSection` that are generalizable to other mesh descriptions.

(sec_petscsection_concept)=

## General concept

Specific entries (or collections of entries) in a `Vec` (or a simple array) can be associated with a "location" on a mesh (or other types of data structure) using the `PetscSection` object.
A **point** is a `PetscInt` that serves as an abstract "index" into arrays from iterable sets, such as k-cells in a mesh.
Other iterable set examples can be as simple as the points of a finite difference grid, or cells of a finite volume grid, or as complex as the topological entities of an unstructured mesh (cells, faces, edges, and vertices).

At it's most basic, a `PetscSection` is a mapping between the mesh points and a tuple `(ndof, offset)`, where `ndof` is the number of values stored at that mesh point and `offset` is the location in the array of that data.
So given the tuple for a mesh point, its data can be accessed by `array[offset + d]`, where `d` in `[0, ndof)` is the dof to access.

### Charts: Defining mesh points

The mesh points for a `PetscSection` must be contiguously numbered and are defined to be in some range $[\mathrm{pStart}, \mathrm{pEnd})$, which is called a **chart**.
The chart of a `PetscSection` is set via `PetscSectionSetChart()`.
Note that even though the mesh points must be contiguously numbered, the indexes into the array (defined by each `(ndof, offset)` tuple) associated with the `PetscSection` need not be.
In other words, there may be elements in the array that are not associated with any mesh points, though this is not often the case.

### Defining the (ndof, offset) tuple

Defining the `(ndof, offset)` tuple for each mesh point generally first starts with setting the `ndof` for each point, which is done using `PetscSectionSetDof()`.
This associates a set of degrees of freedom (dof), (a small space $\{e_k\}\ 0 < k < ndof$), with every point.
If `ndof` is not set for a mesh point, it is assumed to be 0.

The offset for each mesh point is usually set automatically by `PetscSectionSetUp()`.
This will concatenate each mesh point's dofs together in the order of the mesh points.
This concatenation can be done in a different order by setting a permutation, which is described in {any}`sec_petscsection_permutation`.

Alternatively, the offset for each mesh point can be set manually by `PetscSectionSetOffset()`, though this is not commonly needed.

Once the tuples are created, the `PetscSection` is ready to use.

### Basic Setup Example

To summarize, the sequence for constructing a basic `PetscSection` is the following:

1. Specify the range of points, or chart, with `PetscSectionSetChart()`.
2. Specify the number of dofs per point with `PetscSectionSetDof()`. Any values not set will be zero.
3. Set up the `PetscSection` with `PetscSectionSetUp()`.

## Multiple Fields

In many discretizations, it is useful to differentiate between different kinds of dofs present on a mesh.
For example, a dof attached to a cell point might represent pressure while dofs on vertices might represent velocity or displacement.
A `PetscSection` can represent this additional structure with what are called **fields**.
**Fields** are indexed contiguously from `[0, num_fields)`.
To set the number of fields for a `PetscSection`, call `PetscSectionSetNumFields()`.

Internally, each field is stored in a separate `PetscSection`.
In fact, all the concepts and functions presented in {any}`sec_petscsection_concept` were actually applied onto the **default field**, which is indexed as `0`.
The fields inherit the same chart as the "parent" `PetscSection`.

### Setting Up Multiple Fields

Setup for a `PetscSection` with multiple fields is nearly identical to setup for a single field.

The sequence for constructing such a `PetscSection` is the following:

1. Specify the range of points, or chart, with `PetscSectionSetChart()`. All fields share the same chart.
2. Specify the number of fields with `PetscSectionSetNumFields()`.
3. Set the number of dof for each point on each field with `PetscSectionSetFieldDof()`. Any values not set will be zero.
4. Set the **total** number of dof for each point with `PetscSectionSetDof()`. Thus value must be greater than or equal to the sum of the values set with
   `PetscSectionSetFieldDof()` at that point. Again, values not set will be zero.
5. Set up the `PetscSection` with `PetscSectionSetUp()`.

### Point Major or Field Major

A `PetscSection` with one field and and offsets set in `PetscSectionSetUp()` may be thought of as defining a two dimensional array indexed by point in the outer dimension with a variable length inner dimension indexed by the dof at that point: $v[\mathrm{pStart} <= point < \mathrm{pEnd}][0 <= dof < \mathrm{ndof}]$ [^petscsection-footnote].

With multiple fields, this array is now three dimensional, with the outer dimensions being both indexed by mesh points and field points.
Thus, there is a choice on whether to index by points first, or by fields first.
In other words, will the array be laid out in a point-major or field-major fashion.

Point-major ordering corresponds to $v[\mathrm{pStart} <= point < \mathrm{pEnd}][0 <= field < \mathrm{num\_fields}][0 <= dof < \mathrm{ndof}]$.
All the dofs for each mesh point are stored contiguously, meaning the fields are **interlaced**.
Field-major ordering corresponds to $v[0 <= field < \mathrm{num\_fields}][\mathrm{pStart} <= point < \mathrm{pEnd}][0 <= dof < \mathrm{ndof}]$.
The all the dofs for each field are stored contiguously, meaning the points are **interlaced**.

Consider a `PetscSection` with 2 fields and 2 points (from 0 to 2). Let the 0th field have `ndof=1` for each point and the 1st field have `ndof=2` for each point.
Denote each array entry $(p_i, f_i, d_i)$ for $p_i$ being the ith point, $f_i$ being the ith field, and $d_i$ being the ith dof.

Point-major order would result in:

$$
[(p_0, f_0, d_0), (p_0, f_1, d_0), (p_0, f_1, d_1),\\ (p_1, f_0, d_0), (p_1, f_1, d_0), (p_1, f_1, d_1)]
$$

Conversely, field-major ordering would result in:

$$
[(p_0, f_0, d_0), (p_1, f_0, d_0),\\ (p_0, f_1, d_0), (p_0, f_1, d_1), (p_1, f_1, d_0), (p_1, f_1, d_1)]
$$

Note that dofs are always contiguous, regardless of the outer dimensional ordering.

Setting the which ordering is done with `PetscSectionSetPointMajor()`, where `PETSC_TRUE` sets point-major and `PETSC_FALSE` sets field major.

**NOTE:** The current default is for point-major, and many operations on `DMPlex` will only work with this ordering. Field-major ordering is provided mainly for compatibility with external packages, such as LibMesh.

## Working with data

Once a `PetscSection` has been created one can use `PetscSectionGetStorageSize()` to determine the total number of entries that can be stored in an array or `Vec` accessible by the `PetscSection`.
This is most often used when creating a new `Vec` for a `PetscSection` such as:

```
PetscSectionGetStorageSize(s, &n);
VecSetSizes(localVec, n, PETSC_DETERMINE);
VecSetFromOptions(localVec);
```

The memory locations in the associated array are found using an **offset** which can be obtained with:

Single-field `PetscSection`:

```
PetscSectionGetOffset(PetscSection, PetscInt point, PetscInt &offset);
```

Multi-field `PetscSection`:

```
PetscSectionGetFieldOffset(PetscSection, PetscInt point, PetscInt field, PetscInt &offset);
```

The value in the array is then accessed with `array[offset + d]`, where `d` in `[0, ndof)` is the dof to access.

## Global Sections: Constrained and Distributed Data

To handle distributed data and data with constraints, we use a pair of `PetscSections` called the `localSection` and `globalSection`.
Their use for each is described below.

### Distributed Data

`PetscSection` can also be applied to distributed problems as well.
This is done using the same local/global system described in {any}`sec_localglobal`.
To do this, we introduce three new concepts; a `localSection`, `globalSection`, `pointSF`, and `sectionSF`.

Assume the mesh points of the "global" mesh are partitioned amongst processes and that some mesh points are shared between multiple processes (i.e there is an overlap in the partitions).
The shared mesh points define the ghost/halo points needed in many PDE problems.
For each shared mesh point, appoint one process to be the owner of that mesh point.
To describe this parallel mesh point layout, we use a `PetscSF` and call it the `pointSF`.
The `pointSF` describes which processes "own" which mesh points and which process is the owner of each shared mesh point.

Next, for each process define a `PetscSection` that describes the mapping between that process's partition (including shared mesh points) and the data stored on it and call it the `localSection`.
The `localSection` describes the layout of the local vector.
To generate the `globalSection` we use `PetscSectionCreateGlobalSection()`, which takes the `localSection` and `pointSF` as inputs.
The global section returns $-(dof+1)$ for the number of dofs on an unowned (ghost) point, and traditionally $-(off+1)$ for its offset on the owning process.
This behavior of the offsets is controlled via an argument to `PetscSectionCreateGlobalSection()`.
The `globalSection` can be used to create global vectors, just as the local section is used to create local vectors.

To perform the global-to-local and local-to-global communication, we define `sectionSF` to be the `PetscSF` describing the mapping between the local and global vectors.
This is generated via `PetscSFSetGraphSection()`.
Using `PetscSFBcastBegin()` will send data from the global vector to the local vector, while `PetscSFReduceBegin()` will send data from the local vector to the global vector.

If using `DM`, this entire process is done automatically.
The `localSection`, `globalSection`, `pointSF`, and `sectionSF` on a `DM` can be obtained via `DMGetLocalSection()`, `DMGetGlobalSection()`, `DMGetPointSF()`, and `DMGetSectionSF()`, respectively.
Additionally, communication from global to local vectors and vice versa can be done via `DMGlobalToLocal()` and `DMLocalToGlobal()` as described in {any}`sec_localglobal`.
Note that not all `DM` types use this system, such as `DMDA` (see {any}`sec_struct`).

### Constrained Data

In addition to describing parallel data, the `localSection`/`globalSection` pair can be used to describe *constrained* dofs
These constraints usually represent essential (Dirichlet) boundary conditions, or algebraic constraints.
They are dofs that have a given fixed value, so they are present in local vectors for finite element/volume assembly or finite difference stencil application purposes, but generally absent from global vectors since they are not unknowns in the algebraic solves.

Constraints should be indicated in the `localSection`.
Use `PetscSectionSetConstraintDof()` to set the number of constrained dofs for a given point, and `PetscSectionSetConstraintIndices()` to indicate which dofs on the given point are constrained.
This must be done before `PetscSectionCreateGlobalSection()` is called to create the `globalSection`.

Note that it is possible to have constraints set in a `localSection`, but have the `globalSection` be generated to include those constraints.
This is useful when doing some form of post-processing of a solution where you want to access all data (see `DMGetOutputDM()` for example).
See `PetscSectionCreateGlobalSection()` for more details on this.

(sec_petscsection_permutation)=

## Permutation: Changing the order of array data

By default, when `PetscSectionSetUp()` is called, the data laid out in the associated array is assumed to be in the same order of the grid points.
For example, the DoFs associated with grid point 0 appear directly before grid point 1, which appears before grid point 2, etc.

It may be desired to have a different the ordering of data in the array than the order of grid points defined by a section.
For example, one may want grid points associated with the boundary of a domain to appear before points associated with the interior of the domain.

This can be accomplished by either changing the indexes of the grid points themselves, or by informing the section of the change in array ordering.
Either method uses an `IS` to define the permutation.

To change the indices of the grid points, call `PetscSectionPermute()` to generate a new `PetscSection` with the desired grid point permutation.

To just change the array layout without changing the grid point indexing, call `PetscSectionSetPermutation()`.
This must be called before `PetscSectionSetUp()` and will only affect the calculation of the offsets for each grid point.

% TODO: Add example to demonstrate the difference between the two permutation methods

## DMPlex Specific Functionality: Obtaining data from the array

A vanilla `PetscSection` (what's been described up till now) gives a relatively naive perspective on the underlying data; it doesn't describe how DoFs attached to a single grid point are ordered or how different grid points relate to each other.
A `PetscSection` can store and use this extra information in the form of **closures**, **symmetries**, and **closure permutations**.
These features currently target `DMPlex` and other unstructured grid descriptions.
A description of those features will be left to {any}`ch_unstructured`.

```{rubric} Footnotes
```

[^petscsection-footnote]: A `PetscSection` can be thought of as a generalization of `PetscLayout`, in the same way that a fiber bundle is a generalization
    of the normal Euclidean basis used in linear algebra. With `PetscLayout`, we associate a unit vector ($e_i$) with every
    point in the space, and just divide up points between processes.
    Conversely, `PetscSection` associates multiple unit vectors with every mesh point (one for each dof) and divides the mesh points between processes using a `PetscSF` to define the distribution.

```{eval-rst}
.. bibliography:: /petsc.bib
    :filter: docname in docnames
```
