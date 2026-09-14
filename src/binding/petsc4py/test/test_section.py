import unittest

import numpy as np

from petsc4py import PETSc


class TestSectionConstraints(unittest.TestCase):
    def setUp(self):
        self.section = PETSc.Section().create(PETSc.COMM_SELF)
        self.section.setNumFields(1)
        self.section.setChart(0, 2)
        for point in range(2):
            self.section.setDof(point, 3)
            self.section.setFieldDof(point, 0, 3)
            self.section.setConstraintDof(point, 1)
            self.section.setFieldConstraintDof(point, 0, 1)
        self.section.setUp()
        for point in range(2):
            self.section.setConstraintIndices(point, [2])
            self.section.setFieldConstraintIndices(point, 0, [2])

    def tearDown(self):
        self.section.destroy()
        PETSc.garbage_cleanup()

    def testConstraintIndices(self):
        self.section.setConstraintIndices(0, [1])
        self.section.setFieldConstraintIndices(0, 0, [1])
        for point, index in enumerate((1, 2)):
            np.testing.assert_array_equal(
                self.section.getConstraintIndices(point), [index]
            )
            np.testing.assert_array_equal(
                self.section.getFieldConstraintIndices(point, 0), [index]
            )


if __name__ == '__main__':
    unittest.main()
