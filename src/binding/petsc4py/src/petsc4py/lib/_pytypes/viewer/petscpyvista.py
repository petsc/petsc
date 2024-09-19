import pyvista as pv
import numpy as np
from petsc4py import PETSc


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

    def setfromoptions(self, viewer):
        pass

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

    def viewObject(self, viewer, pobj):
        if pobj.klass == 'DM':
            grid = self.convertDMToPV(pobj)
            name = viewer.getFileName()
            if name is None:
              grid.plot(show_edges=True)
            else:
              grid.plot(show_edges=True,off_screen=True,screenshot=name)
        return

    def viewCell(self, grid, c):
        cell = grid.get_cell(c)
        print(cell)
        cell.plot(show_edges=True)
        return
