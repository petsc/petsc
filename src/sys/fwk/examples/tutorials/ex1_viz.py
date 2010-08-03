from petsc4py import PETSc

def Viz(fwk, key, conf, c):
    if c is None:
        # Create a Fwk instance to serve as the component
        c = PETSc.Fwk().create(fwk.comm)
        fwk.registerDependence(key, "DensityField")
        return c
    if conf == "init":
        import numpy
        assert isinstance(c, PETSc.Fwk)
        # Extract a PetscObject with key "DensityField"
        densityField = fwk.getComponent("DensityField")
        # Then extract objects attached to densityField:
        #   a DA and a Vec
        da  = densityField.query("mesh")
        vec = densityField.query("density")
        # Get DA parameters and make sure the Vec conforms to them
        N = numpy.array([0,0,0])
        N[0],N[1],N[2] = da.getSizes()
        dim            = da.getDim()
        dof            = da.getDof()
        v = vec.getArray()
        assert v.size == N[0]*N[1]*N[2]*dof
        # Reshape the Vec's data array to conform to the DA shape
        shape = list(N)
        shape.append(dof)
        v = v.reshape(shape)
        # Plot the first component of the vec over the da
        # Use a contour plot in 3D or surface plot in 2D
        from enthought.mayavi import mlab as mlab
        if dim == 2:
            mlab.surf(v[:,:,1])
        if dim == 3:
            mlab.contour3d(v[:,:,:,1])   
    return c
