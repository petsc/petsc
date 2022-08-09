.. _chapter_unstructured:

DMPlex: Unstructured Grids in PETSc
-----------------------------------

This chapter introduces the ``DMPLEX`` subclass of ``DM``, which allows
the user to handle unstructured grids using the generic ``DM`` interface
for hierarchy and multi-physics. DMPlex was created to remedy a huge
problem in all current PDE simulation codes, namely that the
discretization was so closely tied to the data layout and solver that
switching discretizations in the same code was not possible. Not only
does this preclude the kind of comparison that is necessary for
scientific investigation, but it makes library (as opposed to monolithic
application) development impossible.

Representing Unstructured Grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main advantage of DMPlex in representing topology is that it
treats all the different pieces of a mesh, e.g. cells, faces, edges, and
vertices, in exactly the same way. This allows the interface to be very
small and simple, while remaining flexible and general. This also allows
“dimension independent programming”, which means that the same algorithm
can be used unchanged for meshes of different shapes and dimensions.

All pieces of the mesh are treated as *points*, which are identified by
``PetscInt``\ s. A mesh is built by relating points to other points, in
particular specifying a “covering” relation among the points. For
example, an edge is defined by being covered by two vertices, and a
triangle can be defined by being covered by three edges (or even by
three vertices). In fact, this structure has been known for a long time.
It is a `Hasse Diagram <http://en.wikipedia.org/wiki/Hasse_diagram>`__, which is a
Directed Acyclic Graph (DAG) representing a cell complex using the
covering relation. The graph edges represent the relation, which also
encodes a partially ordered set (poset).

For example, we can encode the doublet mesh as in :numref:`fig_doubletMesh`,

.. figure:: /images/docs/manual/dmplex_doublet_mesh.svg
  :name: fig_doubletMesh

  A 2D doublet mesh, two triangles sharing an edge.

which can also be represented as the DAG in
:numref:`fig_doubletDAG`.

.. figure:: /images/docs/manual/dmplex_doublet_dag.svg
  :name: fig_doubletDAG

  The Hasse diagram for our 2D doublet mesh, expressed as a DAG.

To use the PETSc API, we first consecutively number the mesh pieces. The
PETSc convention in 3 dimensions is to number first cells, then
vertices, then faces, and then edges. In 2 dimensions the convention is
to number faces, vertices, and then edges. The user is free to violate
these conventions. In terms of the labels in
:numref:`fig_doubletMesh`, these numberings are

.. math:: f_0 \mapsto \mathtt{0}, f_1 \mapsto \mathtt{1}, \quad v_0 \mapsto \mathtt{2}, v_1 \mapsto \mathtt{3}, v_2 \mapsto \mathtt{4}, v_3 \mapsto \mathtt{5}, \quad e_0 \mapsto \mathtt{6}, e_1 \mapsto \mathtt{7}, e_2 \mapsto \mathtt{8}, e_3 \mapsto \mathtt{9}, e_4 \mapsto \mathtt{10}

First, we declare the set of points present in a mesh,

.. code-block::

   DMPlexSetChart(dm, 0, 11);

Note that a *chart* here corresponds to a semi-closed interval (e.g
:math:`[0,11) = \{0,1,\ldots,10\}`) specifying the range of indices we’d
like to use to define points on the current rank. We then define the
covering relation, which we call the *cone*, which are also the in-edges
in the DAG. In order to preallocate correctly, we first setup sizes,

.. code-block::

   DMPlexSetConeSize(dm, 0, 3);
   DMPlexSetConeSize(dm, 1, 3);
   DMPlexSetConeSize(dm, 6, 2);
   DMPlexSetConeSize(dm, 7, 2);
   DMPlexSetConeSize(dm, 8, 2);
   DMPlexSetConeSize(dm, 9, 2);
   DMPlexSetConeSize(dm, 10, 2);
   DMSetUp(dm);

and then point values,

.. code-block::

   DMPlexSetCone(dm, 0, [6, 7, 8]);
   DMPlexSetCone(dm, 1, [7, 9, 10]);
   DMPlexSetCone(dm, 6, [2, 3]);
   DMPlexSetCone(dm, 7, [3, 4]);
   DMPlexSetCone(dm, 8, [4, 2]);
   DMPlexSetCone(dm, 9, [4, 5]);
   DMPlexSetCone(dm, 10, [5, 3]);

There is also an API for the dual relation, using
``DMPlexSetSupportSize()`` and ``DMPlexSetSupport()``, but this can be
calculated automatically by calling

.. code-block::

   DMPlexSymmetrize(dm);

In order to support efficient queries, we also want to construct fast
search structures and indices for the different types of points, which
is done using

.. code-block::

   DMPlexStratify(dm);

Data on Unstructured Grids (PetscSection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The strongest links between solvers and discretizations are

-  the layout of data over the mesh,

-  problem partitioning, and

-  ordering of unknowns.

To enable modularity, we encode the operations above in simple data
structures that can be understood by the linear algebra engine in PETSc
without any reference to the mesh (topology) or discretization
(analysis).

Data Layout by Hand
^^^^^^^^^^^^^^^^^^^

Data is associated with a mesh using the ``PetscSection`` object. A
``PetscSection`` can be thought of as a generalization of
``PetscLayout``, in the same way that a fiber bundle is a generalization
of the normal Euclidean basis used in linear algebra. With
``PetscLayout``, we associate a unit vector (:math:`e_i`) with every
point in the space, and just divide up points between processes. Using
``PetscSection``, we can associate a set of dofs, a small space
:math:`\{e_k\}`, with every point, and though our points must be
contiguous like ``PetscLayout``, they can be in any range
:math:`[\mathrm{pStart}, \mathrm{pEnd})`.

The sequence for setting up any ``PetscSection`` is the following:

#. Specify the range of points, or chart,

#. Specify the number of dofs per point, and

#. Set up the ``PetscSection``.

For example, using the mesh from
:numref:`fig_doubletMesh`, we can lay out data for
a continuous Galerkin :math:`P_3` finite element method,

.. code-block::

   PetscInt pStart, pEnd, cStart, cEnd, c, vStart, vEnd, v, eStart, eEnd, e;

   DMPlexGetChart(dm, &pStart, &pEnd);
   DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);   /* cells */
   DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd);   /* edges */
   DMPlexGetHeightStratum(dm, 2, &vStart, &vEnd);   /* vertices, equivalent to DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); */
   PetscSectionSetChart(s, pStart, pEnd);
   for(c = cStart; c < cEnd; ++c)
       PetscSectionSetDof(s, c, 1);
   for(v = vStart; v < vEnd; ++v)
       PetscSectionSetDof(s, v, 1);
   for(e = eStart; e < eEnd; ++e)
       PetscSectionSetDof(s, e, 2);
   PetscSectionSetUp(s);

``DMPlexGetHeightStratum()`` returns all the points of the requested height
in the DAG. Since this problem is in two dimensions the edges are at
height 1 and the vertices at height 2 (the cells are always at height
0). One can also use ``DMPlexGetDepthStratum()`` to use the depth in the
DAG to select the points. ``DMPlexGetDepth(dm,&depth)`` returns the depth
of the DAG, hence ``DMPlexGetDepthStratum(dm,depth-1-h,)`` returns the
same values as ``DMPlexGetHeightStratum(dm,h,)``.

For :math:`P_3` elements there is one degree of freedom at each vertex, 2 along
each edge (resulting in a total of 4 degrees of freedom along each edge
including the vertices, thus being able to reproduce a cubic function)
and 1 degree of freedom within the cell (the bubble function which is
zero along all edges).

Now a PETSc local vector can be created manually using this layout,

.. code-block::

   PetscSectionGetStorageSize(s, &n);
   VecSetSizes(localVec, n, PETSC_DETERMINE);
   VecSetFromOptions(localVec);

though it is usually easier to use the ``DM`` directly, which also
provides global vectors,

.. code-block::

   DMSetLocalSection(dm, s);
   DMGetLocalVector(dm, &localVec);
   DMGetGlobalVector(dm, &globalVec);

A global vector is missing both the shared dofs which are not owned by this process, as well as *constrained* dofs. These constraints are meant to mimic essential boundary conditions, or Dirichlet constraints. They are dofs that have a given fixed value, so they are present in local vectors for assembly purposes, but absent from global vectors since they are never solved for.

We can indicate constraints in a local section using ``PetscSectionSetConstraintDof()``, to set the number of constrained dofs for a given point, and ``PetscSectionSetConstraintIndices()`` which indicates which dofs on the given point are constrained. Once we have this information, a global section can be created using ``PetscSectionCreateGlobalSection()``, and this is done automatically by the ``DM``. A global section returns :math:`-(dof+1)` for the number of dofs on an unowned point, and :math:`-(off+1)` for its offset on the owning process. This can be used to create global vectors, just as the local section is used to create local vectors.

Data Layout using PetscFE
^^^^^^^^^^^^^^^^^^^^^^^^^

A ``DM`` can automatically create the local section if given a description of the discretization, for example using a ``PetscFE`` object. Below we create a ``PetscFE`` that can be configured from the command line. It is a single, scalar field, and is added to the ``DM`` using ``DMSetField()``. When a local or global vector is requested, the ``DM`` builds the local and global sections automatically.

.. code-block::

  DMPlexIsSimplex(dm, &simplex);
  PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe);
  DMSetField(dm, 0, NULL, (PetscObject) fe);
  DMCreateDS(dm);

To get the :math:`P_3` section above, we can either give the option ``-petscspace_degree 3``, or call ``PetscFECreateLagrange()`` and set the degree directly.

Partitioning and Ordering
^^^^^^^^^^^^^^^^^^^^^^^^^

In exactly the same way as in ``MatPartitioning`` or with
``MatGetOrdering()``, the results of a partition using
``PetscPartitionerDMPlexPartition`` or reordering using ``DMPlexPermute`` are encoded in
an ``IS``. However, the graph is not the adjacency graph of the problem
Jacobian, but the mesh itself. Once the mesh is partitioned and
reordered, the data layout from a ``PetscSection`` can be used to
automatically derive a problem partitioning/ordering.

Influence of Variables on One Another
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Jacobian of a problem is intended to represent the influence of some
variable on other variables in the problem. Very often, this influence
pattern is determined jointly by the computational mesh and
discretization. ``DMCreateMatrix`` must compute this pattern when it
automatically creates the properly preallocated Jacobian matrix. In
``DMDA`` the influence pattern, or what we will call variable
*adjacency*, depends only on the stencil since the topology is Cartesian
and the discretization is implicitly finite difference. In DMPlex,
we allow the user to specify the adjacency topologically, while
maintaining good defaults.

The pattern is controlled by two flags. The first flag, ``useCone``,
indicates whether variables couple first to their boundary and then to
neighboring entities, or the reverse. For example, in finite elements,
the variables couple to the set of neighboring cells containing the mesh
point, and we set the flag to ``useCone = PETSC_FALSE``. By constrast,
in finite volumes, cell variables first couple to the cell boundary, and
then to the neighbors, so we set the flag to ``useCone = PETSC_TRUE``.
The second flag, ``useClosure``, indicates whether we consider the
transitive closure of the neighbor relation above, or just a single
level. For example, in finite elements, the entire boundary of any cell
couples to the interior, and we set the flag to
``useClosure = PETSC_TRUE``. By contrast, in most finite volume methods,
cells couple only across faces, and not through vertices, so we set the
flag to ``useClosure = PETSC_FALSE``. However, the power of this method
is its flexibility. If we wanted a finite volume method that coupled all
cells around a vertex, we could easily prescribe that by changing to
``useClosure = PETSC_TRUE``.

Evaluating Residuals
~~~~~~~~~~~~~~~~~~~~

The evaluation of a residual or Jacobian, for most discretizations has
the following general form:

-  Traverse the mesh, picking out pieces (which in general overlap),

-  Extract some values from the solution vector, associated with this
   piece,

-  Calculate some values for the piece, and

-  Insert these values into the residual vector

DMPlex separates these different concerns by passing sets of points,
which are just ``PetscInt``\ s, from mesh traversal routines to data
extraction routines and back. In this way, the ``PetscSection`` which
structures the data inside a ``Vec`` does not need to know anything
about the mesh inside a DMPlex.

The most common mesh traversal is the transitive closure of a point,
which is exactly the transitive closure of a point in the DAG using the
covering relation. In other words, the transitive closure consists of
all points that cover the given point (generally a cell) plus all points
that cover those points, etc. So in 2d the transitive closure for a cell
consists of edges and vertices while in 3d it consists of faces, edges,
and vertices. Note that this closure can be calculated orienting the
arrows in either direction. For example, in a finite element
calculation, we calculate an integral over each element, and then sum up
the contributions to the basis function coefficients. The closure of the
element can be expressed discretely as the transitive closure of the
element point in the mesh DAG, where each point also has an orientation.
Then we can retrieve the data using ``PetscSection`` methods,

.. code-block::

   PetscScalar *a;
   PetscInt     numPoints, *points = NULL, p;

   VecGetArray(u,&a);
   DMPlexGetTransitiveClosure(dm,cell,PETSC_TRUE,&numPoints,&points);
   for (p = 0; p <= numPoints*2; p += 2) {
     PetscInt dof, off, d;

     PetscSectionGetDof(section, points[p], &dof);
     PetscSectionGetOffset(section, points[p], &off);
     for (d = 0; d <= dof; ++d) {
       myfunc(a[off+d]);
     }
   }
   DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &numPoints, &points);
   VecRestoreArray(u, &a);

This operation is so common that we have built a convenience method
around it which returns the values in a contiguous array, correctly
taking into account the orientations of various mesh points:

.. code-block::

   const PetscScalar *values;
   PetscInt           csize;

   DMPlexVecGetClosure(dm, section, u, cell, &csize, &values);
   /* Do integral in quadrature loop */
   DMPlexVecRestoreClosure(dm, section, u, cell, &csize, &values);
   DMPlexVecSetClosure(dm, section, residual, cell, &r, ADD_VALUES);

A simple example of this kind of calculation is in
``DMPlexComputeL2Diff_Plex()`` (`source <../../src/dm/impls/plex/plexfem.c.html#DMComputeL2Diff_Plex>`__).
Note that there is no restriction on the type of cell or dimension of
the mesh in the code above, so it will work for polyhedral cells, hybrid
meshes, and meshes of any dimension, without change. We can also reverse
the covering relation, so that the code works for finite volume methods
where we want the data from neighboring cells for each face:

.. code-block::

   PetscScalar *a;
   PetscInt     points[2*2], numPoints, p, dofA, offA, dofB, offB;

   VecGetArray(u,  &a);
   DMPlexGetTransitiveClosure(dm, cell, PETSC_FALSE, &numPoints, &points);
   assert(numPoints == 2);
   PetscSectionGetDof(section, points[0*2], &dofA);
   PetscSectionGetDof(section, points[1*2], &dofB);
   assert(dofA == dofB);
   PetscSectionGetOffset(section, points[0*2], &offA);
   PetscSectionGetOffset(section, points[1*2], &offB);
   myfunc(a[offA], a[offB]);
   VecRestoreArray(u, &a);

This kind of calculation is used in
`TS Tutorial ex11 <../../src/ts/tutorials/ex11.c.html>`__.

Saving and Loading Data with HDF5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSc allows users to save/load DMPlexs representing meshes,
``PetscSection``\ s representing data layouts on the meshes, and
``Vec``\ s defined on the data layouts to/from an HDF5 file in
parallel, where one can use different number of processes for saving
and for loading.

Saving
^^^^^^

The simplest way to save DM data is to use options for configuration.
This requires only the code

.. code-block::

  DMViewFromOptions(dm, NULL, "-dm_view");
  VecViewFromOptions(vec, NULL, "-vec_view")

along with the command line options

.. code-block:: console

  $ ./myprog -dm_view hdf5:myprog.h5 -vec_view hdf5:myprog.h5::append

Options prefixes can be used to separately control the saving and loading of various fields.
However, the user can have finer grained control by explicitly creating the PETSc objects involved.
To save data to "example.h5" file, we can first create a ``PetscViewer`` of type ``PETSCVIEWERHDF5`` in ``FILE_MODE_WRITE`` mode as:

.. code-block::

   PetscViewer  viewer;

   PetscViewerHDF5Open(PETSC_COMM_WORLD, "example.h5", FILE_MODE_WRITE, &viewer);

As ``dm`` is a DMPlex object representing a mesh, we first give it a *mesh name*, "plexA", and save it as:

.. code-block::

   PetscObjectSetName((PetscObject)dm, "plexA");
   PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_PETSC);
   DMView(dm, viewer);
   PetscViewerPopFormat(viewer);

The ``DMView()`` call is shorthand for the following sequence

.. code-block::

   DMPlexTopologyView(dm, viewer);
   DMPlexCoordinatesView(dm, viewer);
   DMPlexLabelsView(dm, viewer);

If the *mesh name* is not explicitly set, the default name is used.
In the above ``PETSC_VIEWER_HDF5_PETSC`` format was used to save the entire representation of the mesh.
This format also saves global point numbers attached to the mesh points.
In this example the set of all global point numbers is :math:`X = [0, 11)`.

The data layout, ``s``, needs to be wrapped in a ``DM`` object for it to be saved.
Here, we create the wrapping ``DM``, ``sdm``, with ``DMClone()``, give it a *dm name*, "dmA", attach ``s`` to ``sdm``, and save it as:

.. code-block::

   DMClone(dm, &sdm);
   PetscObjectSetName((PetscObject)sdm, "dmA");
   DMSetLocalSection(sdm, s);
   DMPlexSectionView(dm, viewer, sdm);

If the *dm name* is not explicitly set, the default name is to be used.
In the above, instead of using ``DMClone()``, one could also create a new ``DMSHELL`` object to attach ``s`` to.
The first argument of ``DMPlexSectionView()`` is a ``DMPLEX`` object that represents the mesh, and the third argument is a ``DM`` object that carries the data layout that we would like to save.
They are, in general, two different objects, and the former carries a *mesh name*, while the latter carries a *dm name*.
These names are used to construct a group structure in the HDF5 file.
Note that the data layout points are associated with the mesh points, so each of them can also be tagged with a global point number in :math:`X`; ``DMPlexSectionView()`` saves these tags along with the data layout itself, so that, when the mesh and the data layout are loaded separately later, one can associate the points in the former with those in the latter by comparing their global point numbers.

We now create a local vector assiciated with ``sdm``, e.g., as:

.. code-block::

   Vec  vec;

   DMGetLocalVector(sdm, &vec);

After setting values of ``vec``, we name it "vecA" and save it as:

.. code-block::

   PetscObjectSetName((PetscObject)vec, "vecA");
   DMPlexLocalVectorView(dm, viewer, sdm, vec);

A global vector can be saved in the exact same way with trivial changes.

After saving, we destroy the ``PetscViewer`` with:

.. code-block::

   PetscViewerDestroy(&viewer);

The output file "example.h5" now looks like the following:

::

   HDF5 "example.h5" {
   FILE_CONTENTS {
    group      /
    group      /topologies
    group      /topologies/plexA
    group      /topologies/plexA/dms
    group      /topologies/plexA/dms/dmA
    dataset    /topologies/plexA/dms/dmA/order
    group      /topologies/plexA/dms/dmA/section
    dataset    /topologies/plexA/dms/dmA/section/atlasDof
    dataset    /topologies/plexA/dms/dmA/section/atlasOff
    group      /topologies/plexA/dms/dmA/vecs
    group      /topologies/plexA/dms/dmA/vecs/vecA
    dataset    /topologies/plexA/dms/dmA/vecs/vecA/vecA
    group      /topologies/plexA/topology
    dataset    /topologies/plexA/topology/cells
    dataset    /topologies/plexA/topology/cones
    dataset    /topologies/plexA/topology/order
    dataset    /topologies/plexA/topology/orientation
    }
   }

Loading
^^^^^^^

To load data from "example.h5" file, we create a ``PetscViewer``
of type ``PETSCVIEWERHDF5`` in ``FILE_MODE_READ`` mode as:

.. code-block::

   PetscViewerHDF5Open(PETSC_COMM_WORLD, "example.h5", FILE_MODE_READ, &viewer);

We then create a DMPlex object, give it a *mesh name*, "plexA", and load
the mesh as:

.. code-block::

   DMCreate(PETSC_COMM_WORLD, &dm);
   DMSetType(dm, DMPLEX);
   PetscObjectSetName((PetscObject)dm, "plexA");
   PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_PETSC);
   DMLoad(dm, viewer);
   PetscViewerPopFormat(viewer);

where ``PETSC_VIEWER_HDF5_PETSC`` format was again used. The user can have more control by replace the single load call with

.. code-block::

   PetscSF  sfO;

   DMCreate(PETSC_COMM_WORLD, &dm);
   DMSetType(dm, DMPLEX);
   PetscObjectSetName((PetscObject)dm, "plexA");
   PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_PETSC);
   DMPlexTopologyLoad(dm, viewer, &sfO);
   DMPlexCoordinatesLoad(dm, viewer, sfO);
   PetscViewerPopFormat(viewer);

The object returned by ``DMPlexTopologyLoad()``, ``sfO``, is a
``PetscSF`` that pushes forward :math:`X` to the loaded mesh,
``dm``; this ``PetscSF`` is constructed with the global point
number tags that we saved along with the mesh points.

As the DMPlex mesh just loaded might not have a desired distribution,
it is common to redistribute the mesh for a better distribution using
``DMPlexDistribute()``, e.g., as:

.. code-block::

    DM        distributedDM;
    PetscInt  overlap = 1;
    PetscSF   sfDist, sf;

    DMPlexDistribute(dm, overlap, &sfDist, &distributedDM);
    if (distributedDM) {
      DMDestroy(&dm);
      dm = distributedDM;
      PetscObjectSetName((PetscObject)dm, "plexA");
    }
    PetscSFCompose(sfO, sfDist, &sf);
    PetscSFDestroy(&sfO);
    PetscSFDestroy(&sfDist);

Note that the new DMPlex does not automatically inherit the *mesh name*,
so we need to name it "plexA" once again. ``sfDist`` is a ``PetscSF``
that pushes forward the loaded mesh to the redistributed mesh, so, composed
with ``sfO``, it makes the ``PetscSF`` that pushes forward :math:`X`
directly to the redistributed mesh, which we call ``sf``.

We then create a new ``DM``, ``sdm``, with ``DMClone()``, give it
a *dm name*, "dmA", and load the on-disk data layout into ``sdm`` as:

.. code-block::

   PetscSF  globalDataSF, localDataSF;

   DMClone(dm, &sdm);
   PetscObjectSetName((PetscObject)sdm, "dmA");
   DMPlexSectionLoad(dm, viewer, sdm, sf, &globalDataSF, &localDataSF);

where we could also create a new
``DMSHELL`` object instead of using ``DMClone()``.
Each point in the on-disk data layout being tagged with a global
point number in :math:`X`, ``DMPlexSectionLoad()``
internally constructs a ``PetscSF`` that pushes forward the on-disk
data layout to :math:`X`.
Composing this with ``sf``, ``DMPlexSectionLoad()`` internally
constructs another ``PetscSF`` that pushes forward the on-disk
data layout directly to the redistributed mesh. It then
reconstructs the data layout ``s`` on the redistributed mesh and
attaches it to ``sdm``. The objects returned by this function,
``globalDataSF`` and ``localDataSF``, are ``PetscSF``\ s that can
be used to migrate the on-disk vector data into local and global
``Vec``\ s defined on ``sdm``.

We now create a local vector assiciated with ``sdm``, e.g., as:

.. code-block::

   Vec  vec;

   DMGetLocalVector(sdm, &vec);

We then name ``vec`` "vecA" and load the on-disk vector into ``vec`` as:

.. code-block::

   PetscObjectSetName((PetscObject)vec, "vecA");
   DMPlexLocalVectorLoad(dm, viewer, sdm, localDataSF, localVec);

where ``localDataSF`` knows how to migrate the on-disk vector
data into a local ``Vec`` defined on ``sdm``.
The on-disk vector can be loaded into a global vector associated with
``sdm`` in the exact same way with trivial changes.

After loading, we destroy the ``PetscViewer`` with:

.. code-block::

   PetscViewerDestroy(&viewer);

The above infrastructure works seamlessly in distributed-memory parallel
settings, in which one can even use different number of processes for
saving and for loading; a more comprehensive example is found in
`DMPlex Tutorial ex12 <../../src/dm/impls/plex/tutorials/ex12.c.html>`__.

Metric-based mesh adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DMPlex supports mesh adaptation using the *Riemannian metric framework*.
The idea is to use a Riemannian metric space within the mesher. The
metric space dictates how mesh resolution should be distributed across
the domain. Using this information, the remesher transforms the mesh such
that it is a *unit mesh* when viewed in the metric space. That is, the image
of each of its elements under the mapping from Euclidean space into the
metric space has edges of unit length.

One of the main advantages of metric-based mesh adaptation is that it allows
for fully anisotropic remeshing. That is, it provides a means of controlling
the shape and orientation of elements in the adapted mesh, as well as their
size. This can be particularly useful for advection-dominated and
directionally-dependent problems.

See :cite:`Alauzet2010` for further details on metric-based anisotropic mesh
adaptation.

The two main ingredients for metric-based mesh adaptation are an input mesh
(i.e. the DMPlex) and a Riemannian metric. The implementation in PETSc assumes
that the metric is piecewise linear and continuous across elemental boundaries.
Such an object can be created using the routine

.. code-block::

   DMPlexMetricCreate(DM dm, PetscInt f, Vec *metric);

A metric must be symmetric positive-definite, so that distances may be properly
defined. This can be checked using

.. code-block::

   DMPlexMetricEnforceSPD(DM dm, Vec metricIn, PetscBool restrictSizes, PetscBool restrictAnisotropy, Vec metricOut, Vec determinant);

This routine may also be used to enforce minimum and maximum tolerated metric
magnitudes (i.e. cell sizes), as well as maximum anisotropy. These quantities
can be specified using

.. code-block::

   DMPlexMetricSetMinimumMagnitude(DM dm, PetscReal h_min);
   DMPlexMetricSetMaximumMagnitude(DM dm, PetscReal h_max);
   DMPlexMetricSetMaximumAnisotropy(DM dm, PetscReal a_max);

or the command line arguments

.. code-block::

   -dm_plex_metric_h_min <h_min>
   -dm_plex_metric_h_max <h_max>
   -dm_plex_metric_a_max <a_max>


One simple way to combine two metrics is by simply averaging them entry-by-entry.
Another is to *intersect* them, which amounts to choosing the greatest level of
refinement in each direction. These operations are available in PETSc through
the routines

.. code-block::

   DMPlexMetricAverage(DM dm, PetscInt numMetrics, PetscReal weights[], Vec metrics[], Vec metricAvg);
   DMPlexMetricIntersection(DM dm, PetscInt numMetrics, Vec metrics[], Vec metricInt);

However, before combining metrics, it is important that they are scaled in the same
way. Scaling also allows the user to control the number of vertices in the adapted
mesh (in an approximate sense). This is achieved using the :math:`L^p` normalization
framework, with the routine

.. code-block::

   DMPlexMetricNormalize(DM dm, Vec metricIn, PetscBool restrictSizes, PetscBool restrictAnisotropy, Vec metricOut, Vec determinant);

There are two important parameters for the normalization: the normalization order
:math:`p` and the target metric complexity, which is analogous to the vertex count.
They are controlled using

.. code-block::

   DMPlexMetricSetNormalizationOrder(DM dm, PetscReal p);
   DMPlexMetricSetTargetComplexity(DM dm, PetscReal target);

or the command line arguments

.. code-block:: console

   -dm_plex_metric_p <p>
   -dm_plex_metric_target_complexity <target>

Two different metric-based mesh adaptation tools are available in PETSc:

- `Pragmatic <https://meshadaptation.github.io/>`__;

- `Mmg/ParMmg <https://www.mmgtools.org/>`__.

Mmg is a serial package, whereas ParMmg is the MPI parallel version.
Note that surface meshing is not currently supported and that ParMmg
is only in 3D. Mmg and Pragmatic can be used for both 2D and 3D problems.
Pragmatic, Mmg and ParMmg may be specified by the command line arguments

.. code-block::

   -dm_adaptor pragmatic
   -dm_adaptor mmg
   -dm_adaptor parmmg

Once a metric has been constructed, it can be used to perform metric-based
mesh adaptation using the routine

.. code-block::

   DMAdaptMetric(DM dm, Vec metric, DMLabel bdLabel, DMLabel rgLabel, DM dmAdapt);

where ``bdLabel`` and ``rgLabel`` are boundary and interior tags to be
preserved under adaptation, respectively.

.. raw:: html

    <hr>

.. bibliography:: /petsc.bib
    :filter: docname in docnames


