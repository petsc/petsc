import pyvista as pv
import numpy as np
from petsc4py import PETSc

SCALAR = 0
VECTOR = 1

def _convertCell(ctype, cells, nc, off):
    # The VTK conventions are at https://www.princeton.edu/~efeibush/viscourse/vtk.pdf
    if ctype == PETSc.DM.PolytopeType.TETRAHEDRON:
        tmp = cells[off + 1]
        cells[off + 1] = cells[off + 2]
        cells[off + 2] = tmp
    elif ctype == PETSc.DM.PolytopeType.HEXAHEDRON:
        tmp = cells[off + 1]
        cells[off + 1] = cells[off + 3]
        cells[off + 3] = tmp
    elif ctype == PETSc.DM.PolytopeType.TRI_PRISM:
        tmp = cells[off + 4]
        cells[off + 4] = cells[off + 5]
        cells[off + 5] = tmp
    elif ctype == PETSc.DM.PolytopeType.TRI_PRISM_TENSOR:
        tmp = cells[off + 1]
        cells[off + 1] = cells[off + 2]
        cells[off + 2] = tmp
        tmp = cells[off + 4]
        cells[off + 4] = cells[off + 5]
        cells[off + 5] = tmp
    elif ctype == PETSc.DM.PolytopeType.PYRAMID:
        tmp            = cells[off + 1]
        cells[off + 1] = cells[off + 3]
        cells[off + 3] = tmp
    return


VTK_TYPES = {}
VTK_TYPES[PETSc.DM.PolytopeType.POINT] = pv.CellType.VERTEX
VTK_TYPES[PETSc.DM.PolytopeType.SEGMENT] = pv.CellType.LINE
VTK_TYPES[PETSc.DM.PolytopeType.TRIANGLE] = pv.CellType.TRIANGLE
VTK_TYPES[PETSc.DM.PolytopeType.QUADRILATERAL] = pv.CellType.QUAD
VTK_TYPES[PETSc.DM.PolytopeType.TETRAHEDRON] = pv.CellType.TETRA
VTK_TYPES[PETSc.DM.PolytopeType.HEXAHEDRON] = pv.CellType.HEXAHEDRON
VTK_TYPES[PETSc.DM.PolytopeType.TRI_PRISM] = pv.CellType.WEDGE
VTK_TYPES[PETSc.DM.PolytopeType.TRI_PRISM_TENSOR] = pv.CellType.WEDGE
VTK_TYPES[PETSc.DM.PolytopeType.QUAD_PRISM_TENSOR] = pv.CellType.HEXAHEDRON
VTK_TYPES[PETSc.DM.PolytopeType.PYRAMID] = pv.CellType.PYRAMID


class PetscPyVista:
    def setUp(self, viewer):
        pass

    def setFromOptions(self, viewer):
        OptDB = PETSc.Options(viewer.prefix)
        self.swarmField     = OptDB.getString('view_pyvista_swarm_field', 'w_q')
        self.swarmPointSize = OptDB.getInt('view_pyvista_swarm_point_size', 5)
        self.warpFactor     = OptDB.getReal('view_pyvista_warp', 0.0)
        self.clipBounds     = OptDB.getRealArray('view_pyvista_clip', [])
        self.glyphScale     = OptDB.getReal('view_pyvista_glyph_scale', 0.0)

    def view(self, viewer, outviewer):
        pass

    def flush(self, viewer):
        pass

    def convertDMToPV(self, plex):
        cdim = plex.getCoordinateDim()
        vStart, vEnd = plex.getDepthStratum(0)
        cStart, cEnd = plex.getHeightStratum(0)
        conesLength = 0
        # Maybe it will be faster in C?
        # DMPlexGetCellsVertices?
        for c in range(cStart, cEnd):
            conesLength += 1
            closure, ornt = plex.getTransitiveClosure(c)
            for cl in closure:
                if cl >= vStart and cl < vEnd:
                    conesLength += 1
        cells = np.zeros((conesLength), dtype=np.uint32)
        conesLength = 0
        for c in range(cStart, cEnd):
            closure, ornt = plex.getTransitiveClosure(c)
            nc = 0
            off = 1
            for cl in closure:
                if cl >= vStart and cl < vEnd:
                    cells[conesLength] += 1
                    cells[conesLength + off] = cl - vStart
                    nc += 1
                    off += 1
            _convertCell(plex.getCellType(c), cells, nc, conesLength + 1)
            conesLength += off
        celltypes = np.zeros((cEnd - cStart), dtype=np.uint32)
        for c in range(cStart, cEnd):
            celltypes[c] = VTK_TYPES[plex.getCellType(c)]
        points = np.zeros((vEnd - vStart, 3), dtype=np.float32)
        with plex.getCoordinatesLocal().getBuffer() as coords:
            for v in range(vEnd - vStart):
                for d in range(cdim):
                    points[v, d] = coords[v * cdim + d]
        return pv.UnstructuredGrid(cells, celltypes, points)

    def viewPlex(self, viewer, dm, scalars = None):
        grid  = self.convertDMToPV(dm)
        name  = viewer.getFileName()
        ftype = None
        dim   = dm.getDimension()
        if scalars is not None:
            if scalars[2] == 1:
                ftype = SCALAR
            elif scalars[2] == dim:
                ftype = VECTOR
            else:
                raise RuntimeError('Scalars \'%s\' blocksize %d did not match 1 or mesh dim %d' % (scalars[0], scalars[2], dm.getDimension()))
            if scalars[1].shape[0] / scalars[2] == grid.n_cells:
                grid.cell_data[scalars[0]] = scalars[1]
            elif scalars[1].shape[0] / scalars[2] == grid.n_points:
                if dim == 3:
                    grid.point_data[scalars[0]] = scalars[1].reshape(-1, scalars[2])
                else:
                    vecs = np.zeros((scalars[1].shape[0] // scalars[2], 3))
                    vecs[:, 0:2] = scalars[1].reshape(-1, scalars[2])
                    grid.point_data[scalars[0]] = vecs
            else:
                raise RuntimeError('Scalars \'%s\' size %d (%d) did not match sizes for cells (%d) or vertices (%d)' % (scalars[0], scalars[1].shape[0], scalars[2], grid.n_cells, grid.n_points))
            if self.warpFactor > 0.:
                if ftype == SCALAR:
                    grid = grid.warp_by_scalar(factor = self.warpFactor)
                elif ftype == VECTOR:
                    grid = grid.warp_by_vector(factor = self.warpFactor)
        if len(self.clipBounds) == 6:
            grid = grid.clip_box(self.clipBounds)
        elif len(self.clipBounds) == 3:
            grid = grid.clip(self.clipBounds)
        if name is None:
            pl = pv.Plotter()
            if ftype == VECTOR:
                pl.add_mesh(grid, show_edges=True)
                if self.glyphScale > 0.:
                    grid.point_data["magnitudes"] = self.glyphScale * np.linalg.norm(grid.point_data[scalars[0]], axis=1)
                    pl.add_mesh(grid.glyph(orient=scalars[0], scale="magnitudes"))
                else:
                    pl.add_mesh(grid.glyph(orient=scalars[0], scale=scalars[0]))
            elif ftype == SCALAR:
                pl.add_mesh(grid, show_edges=True, scalars=scalars[0])
            else:
                pl.add_mesh(grid, show_edges=True)
            pl.show()
        else:
            grid.plot(show_edges=True, scalars=scalars[0], off_screen=True, screenshot=name)
        return

    def viewSwarm(self, viewer, sw):
        import math
        name = viewer.getFileName()
        spoints = sw.getField('DMSwarmPIC_coor')
        n       = spoints.shape[0]
        bs      = spoints.shape[1]
        points  = np.zeros((n, 3))
        for i in range(n):
            points[i,:bs] = spoints[i,:]
        vpoints = sw.getField('velocity')
        nv      = vpoints.shape[0]
        vbs     = vpoints.shape[1]
        vpoints = vpoints.reshape((nv, vbs))
        wgt   = sw.getField(self.swarmField)
        field = np.zeros((n,))
        for i in range(n):
            field[i] = wgt[i, 0]
        if name is None:
            pl = pv.Plotter()
        else:
            pl = pv.Plotter(off_screen=True)
        pl.add_points(points, scalars=field, render_points_as_spheres=False, point_size=self.swarmPointSize, name="swarm")
        maxF = field.max()
        for i in range(n):
            if maxF <= 0.:
              continue
            if math.fabs(field[i]) < 0.1 * maxF:
              continue
            if math.fabs(vpoints[i,0]) > 0.001:
              continue
            val = 2. * math.fabs(field[i]) / maxF
            pl.add_mesh(pv.Disc(center = points[i,:], normal = (1., 0., 0.), inner = 0.25 * val, outer = val), opacity = 0.2)
        if name is None:
            pl.show()
        else:
            pl.show(interactive=False,screenshot=name)
        sw.restoreField(self.swarmField)
        sw.restoreField('velocity')
        sw.restoreField('DMSwarmPIC_coor')
        return

    def viewObject(self, viewer, pobj):
        if pobj.klass == 'Vec':
          dm = pobj.getDM()
          a = pobj.getArray(readonly=1)
          bs = pobj.getBlockSize()
          self.viewPlex(viewer, dm, scalars = (pobj.name, a, bs))
        elif pobj.klass == 'DM':
            if pobj.type == 'plex':
                self.viewPlex(viewer, pobj)
            elif pobj.type == 'swarm':
                self.viewSwarm(viewer, pobj)
        return

    def viewCell(self, grid, c):
        cell = grid.get_cell(c)
        print(cell)
        cell.plot(show_edges=True)
        return
