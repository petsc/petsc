from petsc4py import PETSc

class Matrix(object):

    def __init__(self):
        pass

    def create(self, mat):
        pass

    def destroy(self, mat):
        pass

    def setFromOptions(self, mat):
        m = PETSc.Options().getString('enable',None)
        if m is not None:
          setattr(self,m,1)
