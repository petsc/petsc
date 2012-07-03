#   
#   Run python pyjsbuild --output . AMSOptions.py to generate the needed HTML and Javascript
#

import pyjd 

import AMSJavascript.AMS_Comm as AMS_Comm
import AMSJavascript.AMS_Memory as AMS_Memory
import AMSJavascript
import AMSJavascript

from pyjamas.ui.RootPanel import RootPanel
from pyjamas.ui.TextArea import TextArea
from pyjamas.ui.Label import Label
from pyjamas.ui.Button import Button
from pyjamas.ui.HTML import HTML
from pyjamas.ui.VerticalPanel import VerticalPanel
from pyjamas.ui.HorizontalPanel import HorizontalPanel
from pyjamas.ui.ListBox import ListBox
from pyjamas.JSONService import JSONProxy
from pyjamas.ui.Tree import Tree
from pyjamas.ui.TreeItem import TreeItem


class AMSJavascriptExample:
    def onModuleLoad(self):
        self.status=Label()
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
        self.panel.add(self.status)
        RootPanel().add(self.panel)
        self.commobj = AMS_Comm()
        self.tree = None
        if AMSJavascript.sent > AMSJavascript.recv: 
           self.status.setText('Press button again: AMSJavascript.sent '+str(AMSJavascript.sent)+' AMSJavascript.recv '+str(AMSJavascript.recv))
           return
        if self.commobj.commname == 'No AMS publisher running':
           self.status.setText(self.commobj.commname)
        else:
           self.status.setText('Memories for AMS Comm: '+str(AMSJavascript.sent)+str(AMSJavascript.recv)+self.commobj.commname)
           result = self.commobj.get_memory_list()
           if self.tree: self.panel.remove(self.tree)
           self.tree = Tree()
           for i in result:
              subtree = TreeItem(i)
              memory = self.commobj.memory_attach(i)
              fields = memory.get_field_list()
              for j in fields:
                 field = memory.get_field_info(j)
                 subtree.addItem(j+' = '+str(field[4]))
              self.tree.addItem(subtree)
              self.panel.add(self.tree)

    def onClick(self, sender):
        self.status.setText('Button pressed')
        if sender == self.buttonupdate:
            self.commobj = AMS_Comm()
            self.status.setText('Updating data: Press Display list button to refesh')
        if sender == self.button:
            if AMSJavascript.sent > AMSJavascript.recv: 
               self.status.setText('Press button again: AMSJavascript.sent '+str(AMSJavascript.sent)+' AMSJavascript.recv '+str(AMSJavascript.recv))
               return
            if self.commobj.commname == 'No AMS publisher running':
               self.status.setText(self.commobj.commname)
            else:
               self.status.setText('Memories for AMS Comm: '+str(AMSJavascript.sent)+str(AMSJavascript.recv)+self.commobj.commname)
               result = self.commobj.get_memory_list()
               if self.tree: self.panel.remove(self.tree)
               self.tree = Tree()
               for i in result:
                  subtree = TreeItem(i)
                  memory = self.commobj.memory_attach(i)
                  fields = memory.get_field_list()
                  for j in fields:
                     field = memory.get_field_info(j)
                     subtree.addItem(j+' = '+str(field[4]))
                  self.tree.addItem(subtree)
                  self.panel.add(self.tree)


if __name__ == '__main__':
    pyjd.setup()
    app = AMSJavascriptExample()
    app.onModuleLoad()
    pyjd.run()

