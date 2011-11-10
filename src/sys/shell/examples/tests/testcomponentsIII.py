from petsc4py import PETSc

class TestIIIA:
    @staticmethod
    def initialize(shell):
        print "Initializing TestIIIA"

    @staticmethod
    def configure(shell):
        print "Configuring TestIIIA"
        
    @staticmethod
    def call(shell, message):
        print "TestIIIA called with message '" + str(message) + "'"
