#
#   Run
#
#     python pyjsbuild --frame=AMSSnoopObjects.html --enable-debug --enable-wrap-calls --output . AMSSnoopObjects.py
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

import AMS


statusbar = 0

boxes  = {}  # The memory and field name for each writable text box created

class AMSSnoopObjects:
    def onModuleLoad(self):
        global statusbar
        statusbar = Label()
        self.button = Button("Display list of all published memories and fields", self)
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
        self.tree = None

    def textboxlistener(self,arg):
      global boxes,statusbar
      statusbar.setText('User changed value in text box to ' + str(arg.getText()) + " " + str(boxes[arg]))
      # the user has changed this value we should send it back to the AMS program
      boxes[arg][2].set_field_info(boxes[arg][1],arg.getText())

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
               if self.tree: self.panel.remove(self.tree)
            else:
               statusbar.setText('Memories for AMS Comm: '+self.commobj.commname)
               result = self.commobj.get_memory_list()
               if self.tree: self.panel.remove(self.tree)
               self.tree = Tree()
               for i in result:
                  if i == "Stack": continue
                  subtree = TreeItem(i)
                  memory = self.commobj.memory_attach(i)
                  fields = memory.get_field_list()
                  if not isinstance(fields,list): fields = [fields]
                  block  = false
                  for j in fields:
                     field = memory.get_field_info(j)
                     if str(field[1]) == 'AMS_READ':
                       if j == "Publish Block":
                         if field[4] == "true": block = true
                       else:
                         subtree.addItem(j+' = '+str(field[4]))
                     else:
                       if j == "Block" and not block: continue
                       PN = HorizontalPanel()
                       PN.add(Label(Text=j+' ='))
                       tb = TextBox(Text=str(field[4]))
                       boxes[tb] = [i,j,memory]
                       tb.addChangeListener(self.textboxlistener)
                       PN.add(tb)
                       subtree.addItem(PN)
                  self.tree.addItem(subtree)
                  self.panel.add(self.tree)

if __name__ == '__main__':
    pyjd.setup()
    app = AMSSnoopObjects()
    app.onModuleLoad()
    pyjd.run()

