.. _chapter_unstructured:

DMPlex: Unstructured Grids in PETSc
-----------------------------------

This chapter introduces the ``DMPLEX`` subclass of ``DM``, which allows
the user to handle unstructured grids using the generic ``DM`` interface
for hierarchy and multi-physics. ``DMPlex`` was created to remedy a huge
problem in all current PDE simulation codes, namely that the
discretization was so closely tied to the data layout and solver that
switching discretizations in the same code was not possible. Not only
does this preclude the kind of comparison that is necessary for
scientific investigation, but it makes library (as opposed to monolithic
application) development impossible.

Representing Unstructured Grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main advantage of ``DMPlex`` in representing topology is that it
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
It is a Hasse Diagram `Hasse Diagram <http://en.wikipedia.org/wiki/Hasse_diagram>`__, which is a
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

::

   DMPlexSetChart(dm, 0, 11);

Note that a *chart* here corresponds to a semi-closed interval (e.g
:math:`[0,11) = \{0,1,\ldots,10\}`) specifying the range of indices we’d
like to use to define points on the current rank. We then define the
covering relation, which we call the *cone*, which are also the in-edges
in the DAG. In order to preallocate correctly, we first setup sizes,

::

   DMPlexSetConeSize(dm, 0, 3);
   DMPlexSetConeSize(dm, 1, 3);
   DMPlexSetConeSize(dm, 6, 2);
   DMPlexSetConeSize(dm, 7, 2);
   DMPlexSetConeSize(dm, 8, 2);
   DMPlexSetConeSize(dm, 9, 2);
   DMPlexSetConeSize(dm, 10, 2);
   DMSetUp(dm);

and then point values,

::

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

::

   DMPlexSymmetrize(dm);

In order to support efficient queries, we also want to construct fast
search structures and indices for the different types of points, which
is done using

::

   DMPlexStratify(dm);

Data on Unstructured Grids
~~~~~~~~~~~~~~~~~~~~~~~~~~

The strongest links between solvers and discretizations are

-  the layout of data over the mesh,

-  problem partitioning, and

-  ordering of unknowns.

To enable modularity, we encode the operations above in simple data
structures that can be understood by the linear algebra engine in PETSc
without any reference to the mesh (topology) or discretization
(analysis).

Data Layout
^^^^^^^^^^^

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

#. Specify the chart,

#. Specify the number of dofs per point, and

#. Set up the ``PetscSection``.

For example, using the mesh from
:numref:`fig_doubletMesh`, we can lay out data for
a continuous Galerkin :math:`P_3` finite element method,

::

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

DMPlexGetHeightStratum() returns all the points of the requested height
in the DAG. Since this problem is in two dimensions the edges are at
height 1 and the vertices at height 2 (the cells are always at height
0). One can also use ``DMPlexGetDepthStratum()`` to use the depth in the
DAG to select the points. ``DMPlexGetDepth(,&depth)`` routines the depth
of the DAG, hence ``DMPlexGetDepthStratum(dm,depth-1-h,)`` returns the
same values as ``DMPlexGetHeightStratum(dm,h,)``.

For P3 elements there is one degree of freedom at each vertex, 2 along
each edge (resulting in a total of 4 degrees of freedom alone each edge
including the vertices, thus being able to reproduce a cubic function)
and 1 degree of freedom within the cell (the bubble function which is
zero along all edges).

Now a PETSc local vector can be created manually using this layout,

::

   PetscSectionGetStorageSize(s, &n);
   VecSetSizes(localVec, n, PETSC_DETERMINE);
   VecSetFromOptions(localVec);

though it is usually easier to use the ``DM`` directly, which also
provides global vectors,

::

   DMSetLocalSection(dm, s);
   DMGetLocalVector(dm, &localVec);
   DMGetGlobalVector(dm, &globalVec);

Partitioning and Ordering
^^^^^^^^^^^^^^^^^^^^^^^^^

In exactly the same way as in ``MatPartitioning`` or with
``MatGetOrdering()``, the results of a partition using
``DMPlexPartition`` or reordering using ``DMPlexPermute`` are encoded in
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
and the discretization is implicitly finite difference. In ``DMPlex``,
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

``DMPlex`` separates these different concerns by passing sets of points,
which are just ``PetscInt``\ s, from mesh traversal routines to data
extraction routines and back. In this way, the ``PetscSection`` which
structures the data inside a ``Vec`` does not need to know anything
about the mesh inside a ``DMPlex``.

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

::

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

::

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

::

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

Networks
~~~~~~~~

Built on top of DMPlex, the DMNetwork subclass provides
abstractions for representing general unstructured networks such as
communication networks, power grid, computer networks, transportation
networks, electrical circuits, graphs, and others.

Application flow
^^^^^^^^^^^^^^^^

The general flow of an application code using DMNetwork is as
follows:

#. Create a network object.

   ::

      DMNetworkCreate(MPI_Comm comm, DM *dm);

#. Create components and register them with the network. A “component”
   is specific application data at a vertex/edge of the network required
   for its residual evaluation. For example, components could be
   resistor/inductor data for circuit applications, edge weights for
   graph problems, or generator/transmission line data for power grids.
   Components are registered by calling

   ::

      DMNetworkRegisterComponent(DM dm, const char *name, size_t size, PetscInt *compkey);

   Here, ``name`` is the component name, ``size`` is the size of
   component data type, and ``compkey`` is an integer key that can be
   used for setting/getting the component at a vertex or an edge.
   DMNetwork currently allows upto 36 components to be registered for a
   network.

#. A DMNetwork can consist of one or more physical subnetworks. When
   multiple physical subnetworks are used one can (optionally) provide
   coupling information between subnetworks which consist only of shared vertices of the physical subnetworks. The
   topological sizes of the network are set by calling

   ::

      DMNetworkSetNumSubNetworks(DM dm, PetscInt nsubnet, PetscInt Nsubnet);

   Here, ``nsubnet`` and ``Nsubnet`` are the local and global number of subnetworks.

#. A subnetwork is added to the network by calling

   ::

      DMNetworkAddSubnetwork(DM dm, const char* name, PetscInt nv, PetscInt ne, PetscInt edgelist[], PetscInt *netnum);

   Here ``name`` is the subnetwork name, ``nv`` and ``ne`` are the numbers of local vertices and local edges on the subnetwork, and ``edgelist`` is the connectivity for the subnetwork.
   The output ``netnum`` is the global numbering of the subnetwork in the network.
   Each element of ``edgelist`` is an integer array of size ``2*ne``
   containing the edge connectivity for the subnetwork.

   | As an example, consider a network comprising of 2 subnetworks that
     are coupled. The topological information for the network is as
     follows:
   | subnetwork 0: v0 — v1 — v2 — v3
   | subnetwork 1: v1 — v2 — v0
   | The two subnetworks are coupled by merging vertex 0 from subnetwork 0 with vertex 2 from subnetwork 1.
   | The ``edgelist`` of this network is
   | edgelist[0] = {0,1,1,2,2,3}
   | edgelist[1] = {1,2,2,0}

   The coupling is done by calling

   ::

      DMNetworkAddSharedVertices(DM dm, PetscInt anet, PetscInt bnet, PetscInt nsv, PetscInt asv[], PetscInt bsv[]);

   Here ``anet`` and ``bnet`` are the first and second subnetwork global numberings returned by ``DMNetworkAddSubnetwork()``,
   ``nsv`` is the number of vertices shared by the two subnetworks, ``asv`` and ``bsv`` are the vertex indices in the subnetwork ``anet`` and ``bnet`` .

#. The next step is to have DMNetwork create a bare layout (graph) of
   the network by calling

   ::

      DMNetworkLayoutSetUp(DM dm);

#. After completing the previous steps, the network graph is set up, but
   no physics is associated yet. This is done by adding the components
   and setting the number of variables to the vertices and edges.

   A component and number of variables are added to a vertex/edge by calling

   ::

      DMNetworkAddComponent(DM dm, PetscInt p, PetscInt compkey, void* compdata, PetscInt nvar)

   where ``p`` is the network vertex/edge point in the range obtained by
   either ``DMNetworkGetVertexRange()``/``DMNetworkGetEdgeRange()``, ``DMNetworkGetSubnetwork()``, or ``DMNetworkGetSharedVertices()``;
   ``compkey`` is the component key returned when registering the component
   (``DMNetworkRegisterComponent()``); ``compdata`` holds the data for the
   component; and ``nvar`` is the number of variables associated to the added component at this network point. DMNetwork supports setting multiple components (max. 36)
   at a vertex/edge. At a shared vertex, DMNetwork currently requires the owner process of the vertex adds all the components and number of variables.

   DMNetwork currently assumes the component data to be stored in a
   contiguous chunk of memory. As such, it does not do any
   packing/unpacking before/after the component data gets distributed.
   Any such serialization (packing/unpacking) should be done by the
   application.

#. Set up network internal data structures.

   ::

      DMSetUp(DM dm);

#. Distribute the network (also moves components attached with
   vertices/edges) to multiple processors.

   ::

      DMNetworkDistribute(DM dm, const char partitioner[], PetscInt overlap, DM *distDM);

#. Associate the ``DM`` with a PETSc solver:

   ::

      KSPSetDM(KSP ksp, DM dm) or SNESSetDM(SNES snes, DM dm) or TSSetDM(TS ts, DM dm).

Utility functions
^^^^^^^^^^^^^^^^^

``DMNetwork`` provides several utility functions for operations on the
network. The mostly commonly used functions are: obtaining iterators for
vertices/edges,

::

   DMNetworkGetEdgeRange(DM dm, PetscInt *eStart, PetscInt *eEnd);

::

   DMNetworkGetVertexRange(DM dm, PetscInt *vStart, PetscInt *vEnd);

::

   DMNetworkGetSubnetwork(DM dm, PetscInt netnum, PetscInt *nv, PetscInt *ne, const PetscInt **vtx, const PetscInt **edge);

checking the status of a vertex,

::

   DMNetworkIsGhostVertex(DM dm, PetscInt p, PetscBool *isghost);

::

   DMNetworkIsSharedVertex(DM dm, PetscInt p, PetscBool *isshared);

and retrieving local/global indices of vertex/edge component variables for
inserting elements in vectors/matrices,

::

   DMNetworkGetLocalVecOffset(DM dm, PetscInt p, PetscInt compnum, PetscInt *offset);

::

   DMNetworkGetGlobalVecOffset(DM dm, PetscInt p, PetscInt compnum, PetscInt *offsetg).

In network applications, one frequently needs to find the supporting
edges for a vertex or the connecting vertices covering an edge. These
can be obtained by the following two routines.

::

   DMNetworkGetConnectedVertices(DM dm, PetscInt edge, const PetscInt *vertices[]);

::

   DMNetworkGetSupportingEdges(DM dm, PetscInt vertex, PetscInt *nedges, const PetscInt *edges[]).

Retrieving components and number of variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The components and the corresponding number of variables set at a vertex/edge can be accessed by

::

   DMNetworkGetComponent(DM dm, PetscInt p, PetscInt compnum, PetscInt *compkey, void **component, PetscInt *nvar)

input ``compnum`` is the component number, output ``compkey`` is the key set by ``DMNetworkRegisterComponent``. An example
of accessing and retrieving the components and number of variables at vertices is:

::

   PetscInt Start,End,numcomps,key,v,compnum;
   void *component;

   DMNetworkGetVertexRange(dm, &Start, &End);
   for (v = Start; v < End; v++) {
     DMNetworkGetNumComponents(dm, v, &numcomps);
     for (compnum=0; compnum < numcomps; compnum++) {
       DMNetworkGetComponent(dm, v, compnum, &key, &component, &nvar);
       compdata = (UserCompDataType)(component);
     }
   }

The above example does not explicitly use the component key. It is
used when different component types are set at different vertices. In
this case, ``compkey`` is used to differentiate the component type.
