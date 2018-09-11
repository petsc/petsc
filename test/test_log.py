# --------------------------------------------------------------------

if __name__ == "__main__":
    import sys, petsc4py
    petsc4py.init(sys.argv+['-log_summary'])

# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class TestLog(unittest.TestCase):

    def setUp(self):
        #PETSc.Log.begin()
        # register stages
        self.stage1 = PETSc.Log.Stage('Stage 1')
        self.stage2 = PETSc.Log.Stage('Stage 2')
        # register classes
        self.klassA = PETSc.Log.Class('Class A')
        self.klassB = PETSc.Log.Class('Class B')
        # register events
        self.event1 = PETSc.Log.Event('Event 1') # no class
        self.event2 = PETSc.Log.Event('Event 2') # no class
        self.eventA = PETSc.Log.Event('Event A', self.klassA)
        self.eventB = PETSc.Log.Event('Event B', self.klassB)

    def testGetName(self):
        self.assertEqual(self.klassA.name, 'Class A')
        self.assertEqual(self.klassB.name, 'Class B')
        self.assertEqual(self.event1.name, 'Event 1')
        self.assertEqual(self.event2.name, 'Event 2')
        self.assertEqual(self.eventA.name, 'Event A')
        self.assertEqual(self.eventB.name, 'Event B')
        self.assertEqual(self.stage1.name, 'Stage 1')
        self.assertEqual(self.stage2.name, 'Stage 2')

    def testLogBeginEnd(self):
        # -----
        self._run_events() # in main stage
        self._run_stages() # in user stages
        # -----
        for event in self._get_events():
            event.deactivate()
            event.setActive(False)
            event.active = False
        self._run_events() # should not be logged
        for event in self._get_events():
            event.activate()
            event.setActive(True)
            event.active = True
        # -----
        for klass in self._get_classes():
            klass.deactivate()
            klass.setActive(False)
            klass.active = False
        self._run_events() # A and B should not be logged
        for klass in self._get_classes():
            klass.activate()
            klass.setActive(True)
            klass.active = True
        # -----
        for stage in self._get_stages():
            active = stage.getActive()
            self.assertTrue(active)
            self.assertTrue(stage.active)
            stage.setActive(False)
            active = stage.getActive()
            self.assertFalse(active)
            self.assertFalse(stage.active)
        self._run_stages() # should not be logged
        for stage in self._get_stages():
            stage.setActive(True)
            stage.active = True
            active = stage.getActive()
            self.assertTrue(active)
            self.assertTrue(stage.active)
        # -----
        self._run_events()
        self._run_stages()

    def _run_stages(self):
        for stage in self._get_stages():
            self._run_events(stage)

    def _get_stages(self):
        return (self.stage1, self.stage2)

    def _get_classes(self):
        return (self.klassA, self.klassB)

    def _get_events(self):
        return (self.event1, self.event2,
                self.eventA, self.eventB)

    def _run_events(self, stage=None):
        if stage is not None:
            stage.push()
        self._events_begin()
        self._events_end()
        if stage is not None:
            stage.pop()

    def _events_begin(self):
        for event in self._get_events():
            event.begin()

    def _events_end(self):
        for event in reversed(self._get_events()):
            event.end()


# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

