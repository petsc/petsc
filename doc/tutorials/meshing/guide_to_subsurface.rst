==================================================
Guide to the Meshing for Subsurface Flows in PETSc
==================================================

This tutorials guides users in creating meshes for the TDyCore simulation framework for subsurface flows. The user inputs a surface mesh, a refinement prescription, and an extrusion prescription in order to create the simulation mesh.

Reading the ASCII Output
------------------------

For example, a very simple mesh would start with a square surface mesh divided into two triangles, which is then extruded to form two triangular prisms. This is the first test in the DMPlex tutorial code ex10,

.. code-block:: console

  $ make -f ./gmakefile test globsearch="dm_impls_plex_tutorials-ex10_0"

which outputs

.. literalinclude:: /../src/dm/impls/plex/tutorials/output/ex10_0.out

We can see that there are two 3-cells, meaning three-dimensional cells, and from the `celltype` label we see that those cells have celltype 9, meaning they are triangular prisms. The original surface mesh had 5 edges, so we would expect 10 edges for the two surfaces and four edges connecting those surfaces. This is exactly what we see, since there are 14 1-cells, but 4 of them noted in parentheses are tensor cells created by extrusion. We can see this another way in the celltype label, where there are ten mesh points of type 1, meaning segments, and four mesh points of type 2, meaning tensor products of a vertex and segment. Similarly, there are 9 2-cells, but 5 of them stretch between the two surfaces, meaning they are tensor products of two segments.

Refinement of Simplex Meshes
----------------------------

In PETSc, we can refine meshes uniformly and adaptively. Adaptive refinement can be controlled using tagging of cells, as well as specifying a target metric for the refined mesh. We will focus on the first type of adaptivity in this tutorial.

Regular Refinement
^^^^^^^^^^^^^^^^^^

We can regularly refine the surface before extrusion using `-dm_refine <k>`, where `k` is the number of refinements,

.. code-block:: console

  $ make -f ./gmakefile test globsearch="dm_impls_plex_tutorials-ex10_1" EXTRA_OPTIONS="-srf_dm_refine 2 -srf_dm_view draw -draw_save $PETSC_DIR/surface.png -draw_save_single_file"

which produces the following surface

.. figure:: /images/tutorials/meshing/surface.png
   :align: center

   **Surface mesh refined twice**

and the extruded mesh can be visualized using VTK. Here I make the image using Paraview, and give the extrusion 3 layers

.. code-block:: console

  $ make -f ./gmakefile test globsearch="dm_impls_plex_tutorials-ex10_1" EXTRA_OPTIONS="-dm_view hdf5:$PETSC_DIR/mesh.h5 -dm_extrude 3"
  $ $PETSC_DIR/lib/petsc/bin/petsc_gen_xmdf.py mesh.h5

.. figure:: /images/tutorials/meshing/extrusion.png
   :align: center

   **Extruded mesh with refined surface**

We can similarly look at this in parallel. Test 2 uses three refinements and three extrusion layers on five processes

.. code-block:: console

  $ make -f ./gmakefile test globsearch="dm_impls_plex_tutorials-ex10_2" EXTRA_OPTIONS="-dm_view hdf5:$PETSC_DIR/mesh.h5 -dm_partition_view -petscpartitioner_type parmetis"
  $ $PETSC_DIR/lib/petsc/bin/petsc_gen_xmdf.py mesh.h5

.. figure:: /images/tutorials/meshing/extrusionParallel.png
   :align: center

   **Parallel extruded mesh with refined surface**

Adaptive Refinement
^^^^^^^^^^^^^^^^^^^

Adaptive refinement of simplicial meshes is somewhat tricky when we demand that the meshes be conforming, as we do in this case. We would like different grid cells to have different levels of refinement, for example headwaters cells in a watershed be refined twice, while river channel cells be refined four times. In order to differentiate between cells, we first mark the cells on the surface using a `DMLabel`. We can do this programmatically,

.. literalinclude:: /../src/dm/impls/plex/tutorials/ex10.c
   :start-at: static PetscErrorCode CreateDomainLabel(
   :end-at: PetscFunctionReturn(0);
   :append: }

or you can label the mesh using a GUI, such as GMsh, and PETSc will read the label values from the input file.

We next create a label marking each cell in the mesh with an action, such as `DM_ADAPT_REFINE` or `DM_ADAPT_COARSEN`. We do this based on a volume constraint, namely that cells with a certain label value should have a certain volume. You could, of course, choose a more complex strategy, but here we just want a clear criterion. We can give volume constraints for label value `v` using the command line argument `-volume_constraint_<v> <vol>`. The mesh is then refined iteratively, checking the volume constraints each time,

.. literalinclude:: /../src/dm/impls/plex/tutorials/ex10.c
   :start-at: while (adapt) {
   :end-at: CHKERRQ(DMLabelDestroy(&adaptLabel));
   :append: }

Test 3 from `ex10` constrains the headwater cells (with marker 1) to have volume less than 0.01, and the river channel cells (with marker 2) to be smaller than 0.000625

.. literalinclude:: /../src/dm/impls/plex/tutorials/ex10.c
   :start-at: suffix: 3
   :lines: 1-3

We can look at a parallel run using extra options for the test system

.. code-block:: console

  $ make -f ./gmakefile test globsearch="dm_impls_plex_tutorials-ex10_3" EXTRA_OPTIONS="-dm_view hdf5:$PETSC_DIR/mesh.h5 -dm_partition_view -petscpartitioner_type parmetis" NP=5
  $ $PETSC_DIR/lib/petsc/bin/petsc_gen_xmdf.py mesh.h5

.. figure:: /images/tutorials/meshing/extrusionAdaptiveParallel.png
   :align: center

   **Parallel extruded mesh with adaptively refined surface**

By turning on `PetscInfo`, we can see what decisions the refiner is making

.. code-block:: console

  $ make -f ./gmakefile test globsearch="dm_impls_plex_tutorials-ex10_3" EXTRA_OPTIONS="-info :dm"
  #       > [0] AdaptMesh(): Adapted mesh, marking 12 cells for refinement, and 0 cells for coarsening
  #       > [0] AdaptMesh(): Adapted mesh, marking 29 cells for refinement, and 0 cells for coarsening
  #       > [0] AdaptMesh(): Adapted mesh, marking 84 cells for refinement, and 0 cells for coarsening
  #       > [0] AdaptMesh(): Adapted mesh, marking 10 cells for refinement, and 0 cells for coarsening
