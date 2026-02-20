(ch_dmcommonality)=

# DM Commonalities

We have introduced a variety of seemingly very different `DM`. Here, we will try to explore the commonalities between them to emphasize that despite superficial
differences they all tackle the three same basic problems:

1. How to map between the indices of (geometric) entities, in some physical space such as $R^3$ or perhaps an abstract space, and
  the storage location (offsets) of numerical values associated with said entities in PETSc vectors or arrays.
  In PETSc, these geometric entities are referred to as points.
2. How to iterate over the entities of interest (for example, points) to produce updates to the numerical values associated with the entities in
  the PETSc vectors or arrays.
3. How to store/access the "connectivity information" that is needed at each entity (point) and provides the correct data dependencies
   needed to compute the numerical updates.

For several `DM` we will devote a short paragraph for each of the three problems.

## DMDA simple structured grids

For structured grids, {any}`sec_struct`, the indexing is trivial, the points are represented as tuples $(i, j, k)$, where $l_i \le i \le u_i$, $l_j \le j \le u_j$,
and $l_k \le k \le u_k.$ `DMDAVecGetArray()` returns a multidimensional array that trivially provides the mapping from said points to the numerical values.
Note that when the programming language gives access to the values in a multi-dimensional array,
internally it computes the `offset` from the beginning of the array using a formula based on the value of $i$, $j$, and $k$ and the array dimensions.

To iterate over the local points, one uses `DMDAGetCorners()` or `DMDAGetGhostCorners()` and iterates over the tuples within the bounds. Specific points, for example
boundary points, can be skipped or processed differently based on the index values.

For finite difference methods on structured grids using a stencil formula, the "connectivity information" is defined implicitly by the stencil needed
by the given discretization and is the same for all grid points (except maybe boundaries or other special points). For example, for the standard
seven point stencil computed at the $(i, j, k)$ entity one needs the numerical values at the $(i \pm 1, j, k)$, $(i, j  \pm 1, k)$,
and $(i, j, k \pm 1)$ entities.

## DMSTAG simple stagger grids

A staggered grid, {any}`ch_stag`, extends the idea of a simple structured grid by allowing not only entities associated with grid vertices (or equivalently cells)
as with `DMDA`
but also with grid edges, grid faces, and grid cells (also called elements). As with `DMDA` each type of entity must have the same number of associated
numerical values. As with simple structured grids, each cell can be represented as a $(i, j, k)$ tuple. But, in addition we need to represent what vertex,
edge, or face of the cell we are referring to. This is done using `DMStagStencilLocation`. `DMStagVecGetArray()` returns a multidimensional array indexed
by $(i, j, k)$ plus a `slot` that tells us which entity on the cell is being accessed. Since a staggered grid can have any problem-dependent
number of numerical values associated with a given entity type, the function `DMStagGetLocationSlot()` provides the final index needed for the array access. After
this the programming language than computes the `offset` from the beginning of the array from the provided indices using a simple formula.

To iterate over the local points, one uses `DMStagGetCorners()` or `DMStagGetGhostCorners()` and iterates over the tuples within the bounds with an inner iteration
of the point entities desired for the application. For example, for a discretization with cell-centered pressures and edge-based velocity the application
would process each of these entities.

For finite difference methods on staggered structured grids using a stencil formula the "connectivity information" is again defined implicitly by the stencil needed
by the given discretization and is the same for all grid points (except maybe boundaries or other special points). In addition, any required cross coupling between
different entities needs to be encoded for the given problem. For example, how do the velocities affect the pressure equation. This information is generally
embedded directly in lines of code that implement the finite difference formula and is not represented in the data structures.


## DMPLEX unstructured meshes

For general unstructured grids, {any}`ch_unstructured`, there is no formula for computing the `offset` into the numerical values
for an entity on the grid from its `point` value, since each `point` can have any number of values which is determined at runtime based
 on the grid, PDE, and discretization. Hence all the offsets
must be managed by `DMPLEX`. This is the job of `PetscSection`, {any}`ch_petscsection`. The process of building a `PetscSection` computes
and stores the offsets and then using the `PetscSection` gives access to the needed offsets.

For unstructured grids, one does not in general iterate over all the entities on all the points. Rather it iterates over a subset of the points representing
a particular entity. For example, when using the finite element method, the application iterates all the points representing elements (cells) using
`DMPlexGetHeightStratum()` to access the chart (beginning and end indices) of the cell entities. Then one uses an associated `PetscSection` to determine the offsets into
vectors or arrays for said points. If needed one can then have an inner iteration over the fields associated with the cell.

For `DMPLEX`, the connectivity information is defined by a graph (and stored explicitly in a data structure used to store graphs),
and the connectivity of a point (entity) is obtained by `DMPlexGetTransitiveClosure()`.


## DMNETWORK computations on graphs of nodes and connecting edges

For networks, {any}`ch_network`, the entities are nodes and edges that connect two nodes, each of which can have any number of submodels.
Again, in general, there is no formula that can produce the
appropriate `offset` into the numerical values for a given `point` (node or edge) directly. The routines `DMNetworkGetLocalVecOffset()`
and `DMNetworkGetGlobalVecOffset()`
are used to obtain the needed offsets from a given point (node or edge) and submodel at that point. Internally a `PetscSection` is used to
manage the storage of the `offset` information but the user-level API does not refer to `PetscSection` directly, rather one thinks about
a collection of submodels at each node and edge of the graph.

To iterate over graph vertices (nodes) one uses `DMNetworkGetVertexRange()` to provide its chart (the starting and end indices)
and `DMNetworkGetEdgeRange()` to provide the chart of the edges. One can then iterate over the models on each point
To iterate over sub-networks one can call `DMNetworkGetSubnetwork()` for each network which
returns lists of the vertex and edge points in said network.

For `DMNETWORK`, the connectivity information is defined by a graph, which is is query-able at each entity by
`DMNetworkGetSupportingEdges()` and `DMNetworkGetConnectedVertices()`.

# Is it a programming language issue?

Regarding problem 1. Does the need for these various approaches for mapping between the entities and the related array offsets and the large amount of code
(in particular `PetscSection` come from the limitation of programming languages when working
with complex multidimensional jagged arrays. Both in constructing such arrays at runtime, that is supplying all the jagged information which depends
on the exact problem, and then providing simple syntax to produce the correct offset into the
memory for accessing the numerical values when the simple array access methods do not work.






