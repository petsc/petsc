from petsc4py import PETSc

class MySNES(object):

    def view(self, snes, vwr):
        viewctx = snes.appctx
        if viewctx is not None:
            viewctx = 'C pointer'
        vwr.printfASCII(f'  My context: {viewctx}\n')
