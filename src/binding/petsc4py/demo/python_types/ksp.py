from petsc4py import PETSc


# The user-defined Python class implementing preconditioned Richardson iteration.
class Richardson:
    def __init__(self):
        self.omega = 1.0
        self.work = []

    def create(self, ksp):
        # The default loop measures the norm of b - A x.
        ksp.setPCSide(PETSc.PC.Side.LEFT)
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

    def setFromOptions(self, ksp):
        options = PETSc.Options(ksp.getOptionsPrefix())
        self.omega = options.getReal('ksp_richardson_scale', self.omega)

    def view(self, ksp, viewer):
        if viewer.getType() == PETSc.Viewer.Type.ASCII:
            viewer.printfASCII(f'  relaxation factor: {self.omega:g}\n')

    def setUp(self, ksp):
        self.reset(ksp)
        self.work = ksp.getWorkVecs(right=2)

    def reset(self, ksp):
        for vec in self.work:
            vec.destroy()
        self.work = []

    def destroy(self, ksp):
        self.reset(ksp)

    def step(self, ksp, b, x):
        A, _ = ksp.getOperators()
        z, r = self.work
        A.mult(x, r)
        r.aypx(-1, b)
        ksp.getPC().apply(r, z)
        x.axpy(self.omega, z)
