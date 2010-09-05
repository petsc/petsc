from petsc4py import PETSc

class TestIIIA:
    @staticmethod
    def initialize(fwk):
        print "Initializing TestIIIA"

    @staticmethod
    def configure(fwk):
        print "Configuring TestIIIA"
        
    @staticmethod
    def call(fwk, message):
        print "TestIIIA called with message '" + str(message) + "'"
