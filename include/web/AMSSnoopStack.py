#
#   Run
#
#     python pyjsbuild --frame=AMSSnoopStack.html --enable-debug --enable-wrap-calls --output . AMSSnoopStack.py
#
#   to generate the needed HTML and Javascript
#

import pyjd

from pyjamas.ui.RootPanel import RootPanel
from pyjamas.ui.TextArea import TextArea
from pyjamas.ui.Label import Label
from pyjamas.ui.Button import Button
from pyjamas.ui.HTML import HTML
from pyjamas.ui.VerticalPanel import VerticalPanel
from pyjamas.ui.HorizontalPanel import HorizontalPanel
from pyjamas.ui.ListBox import ListBox
from pyjamas.ui.Tree import Tree
from pyjamas.ui.TreeItem import TreeItem
from pyjamas.ui.TextBox import TextBox
from pyjamas.ui.TextArea import TextArea
import json
import AMS


statusbar = 0

class AMSSnoopStack:
    def onModuleLoad(self):
        global statusbar
        statusbar = Label()
        self.button = Button("Display Current Stack Frames", self)
        self.buttonupdate = Button("Update data from AMS publisher", self)

        buttons = HorizontalPanel()
        buttons.add(self.button)
        buttons.add(self.buttonupdate)
        buttons.setSpacing(8)

        info = """<p>This example demonstrates the calling of the Memory Snooper in PETSc with Pyjamas and <a href="http://json-rpc.org/">JSON-RPC</a>.</p>"""

        self.panel = VerticalPanel()
        self.panel.add(HTML(info))
        self.panel.add(buttons)
        self.panel.add(statusbar)
        RootPanel().add(self.panel)
        self.commobj = AMS.AMS_Comm()
        self.textarea = None

    def onClick(self, sender):
        global statusbar,boxes
        statusbar.setText('Button pressed')
        pass
        if sender == self.buttonupdate:
            self.commobj = AMS.AMS_Comm()
            statusbar.setText('Updating data: Press Display list button to refesh')
        if sender == self.button:
            if AMS.sent > AMS.recv:
               statusbar.setText('Press button again: sent '+str(AMS.sent)+' recv '+str(AMS.recv))
            if self.commobj.commname == 'No AMS publisher running' or not self.commobj.commname or  self.commobj.comm == -1:
               if self.textarea: self.panel.remove(self.textarea)
               pass
            else:
               statusbar.setText('Memories for AMS Comm: '+self.commobj.commname)
               result = self.commobj.get_memory_list()
               if self.textarea: self.panel.remove(self.textarea)
               self.textarea = TextArea()
               memory = self.commobj.memory_attach("Stack")
               size = memory.get_field_info("current size")
               functions = memory.get_field_info("functions")
               funcs = '\n'.join(functions[4])
               self.textarea.setText(str(funcs))
               self.textarea.setVisibleLines(size[4])
               self.panel.add(self.textarea)

if __name__ == '__main__':
    pyjd.setup()
    app = AMSSnoopStack()
    app.onModuleLoad()
    pyjd.run()

