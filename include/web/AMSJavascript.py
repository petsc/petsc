#
#   Run
#
#     python pyjsbuild --frame=AMSJavascript.html --enable-debug --enable-wrap-calls --output . AMSJavascript.py
#     hg revert AMSJavascript.html
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
from pyjamas.JSONService import JSONProxy
from pyjamas.ui.Tree import Tree
from pyjamas.ui.TreeItem import TreeItem
from pyjamas.ui.TextBox import TextBox


statusbar = 0
args   = {}  # Arguments to each remote call
sent   = 0   # Number of calls sent to server
recv   = 0   # Number of calls received from server
boxes  = {}  # The memory and field name for each writable text box created

class AMSJavascriptExample:
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
        self.commobj = AMS_Comm()
        self.tree = None

    def textboxlistener(self,arg):
      global boxes,statusbar
      statusbar.setText('User changed value in text box to ' + str(arg.getText()) + " " + str(boxes[arg]))
      # the user has changed this value we should send it back to the AMS program
      boxes[arg][2].set_field_info(boxes[arg][1],arg.getText())

    def onClick(self, sender):
        global statusbar,sent,recv,boxes
        statusbar.setText('Button pressed')
        pass
        if sender == self.buttonupdate:
            self.commobj = AMS_Comm()
            if self.commobj.commname == 'No AMS publisher running' or not self.commobj.commname or  self.commobj.comm == -1:
              if self.tree: self.panel.remove(self.tree)
              statusbar.setText('Publisher is not accessable')
            else:
              statusbar.setText('Updating data: Press Display list button to refesh')
        if sender == self.button:
            if sent > recv:
               statusbar.setText('Press button again: sent '+str(sent)+' recv '+str(recv))
            if self.commobj.commname == 'No AMS publisher running' or not self.commobj.commname or  self.commobj.comm == -1:
               if self.tree: self.panel.remove(self.tree)
            else:
               statusbar.setText('Memories for AMS Comm: '+self.commobj.commname)
               result = self.commobj.get_memory_list()
               if self.tree: self.panel.remove(self.tree)
               self.tree = Tree()
               for i in result:
                  subtree = TreeItem(i)
                  memory = self.commobj.memory_attach(i)
                  fields = memory.get_field_list()
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


class ServicePython(JSONProxy):
    def __init__(self):
        JSONProxy.__init__(self, "No service name", ["YAML_echo", "YAML_AMS_Connect", "YAML_AMS_Comm_attach", "YAML_AMS_Comm_get_memory_list","YAML_AMS_Memory_attach","YAML_AMS_Memory_get_field_list","YAML_AMS_Memory_get_field_info","YAML_AMS_Memory_set_field_info","YAML_AMS_Memory_update_send_begin"])

# ---------------------------------------------------------
class AMS_Memory(JSONProxy):
    def __init__(self,comm,memory):
        global args,sent,recv
        self.comm             = comm
        self.name             = memory   # string name of memory
        self.fieldlist        = []
        self.fields           = {}
        self.remote           = ServicePython()
        id = self.remote.YAML_AMS_Memory_attach(comm,memory,self)
        args[id] = ['YAML_AMS_Memory_attach',comm,memory]
        sent += 1

    def get_field_list(self,func = null):
        '''If called with func (net yet supported) then calls func asynchronously with latest memory list;
           otherwise returns current (possibly out-dated) memory list'''
        return self.fieldlist

    def get_field_info(self,field, func = null):
        '''Pass in string name of AMS field
           If called with func (not yet done) then first updates comm with latest field list and then calls func with field'''
        if not self.fields.has_key(field):
            return 'Memory does not have field named '+field
        return self.fields[field]

    def set_field_info(self,field,value,funct = null):
        '''Pass in string name of AMS field and value to be set back on publisher
           If called with func (not yet done) then first updates remove and then calls func with any error information'''
        if not self.fields.has_key(field):
            return 'Memory does not have field named '+field+' thus value not set'
        id = self.remote.YAML_AMS_Memory_set_field_info(self.memory,field,value,self)
        args[id] = ['YAML_AMS_Memory_set_field_info',comm,memory,field]

    def update_send_begin(self,funct = null):
        '''Tells the accessor to update the values in the memory on the publisher
           If called with func (not yet done) then first updates remove and then calls func with any error information'''
        id = self.remote.YAML_AMS_Memory_update_send_begin(self.memory,self)

    def onRemoteResponse(self, response, request_info):
        global args,sent,recv,statusbar
        recv += 1
        method = str(request_info.method)
        rid    = request_info.id
        if method == "YAML_AMS_Memory_attach":
            self.memory = response[0]
            id = self.remote.YAML_AMS_Memory_get_field_list(self.memory,self)
            args[id] = ['YAML_AMS_Memory_get_field_list',self.memory]
            sent += 1
        elif method == "YAML_AMS_Memory_get_field_list":
            self.fieldlist = response
            for i in self.fieldlist:
                id = self.remote.YAML_AMS_Memory_get_field_info(self.memory,i,self)
                args[id] = ['YAML_AMS_Memory_get_field_info',self.memory,i]
                sent += 1
        elif method == "YAML_AMS_Memory_get_field_info":
            self.fields[args[rid][2]] = response
        elif method == "YAML_AMS_Memory_set_field_info":
          self.update_send_begin()
          statusbar.setText("Value updated on server")


    def onRemoteError(self, code, errobj, request_info):
        global statusbar,recv
        recv += 1
        method = str(request_info.method)
        if method == "YAML_AMS_Memory_update_send_begin":
          self.comm.comm = -1;
          statusbar.setText("Publisher is no longer accessable")
        else:
          statusbar.setText("Error "+str(errobj))

# ---------------------------------------------------------
class AMS_Comm(JSONProxy):
    def __init__(self):
        global args,sent,recv
        self.comm             = -1
        self.commname         = ''
        self.memlist          = []
        self.memories         = {}
        self.memory_list_func = null
        self.remote           = ServicePython()
        id = self.remote.YAML_AMS_Connect('No argument', self)
        args[id] = ['YAML_AMS_Connect']
        sent += 1

    def get_memory_list(self,func = null):
        '''If called with func then calls func asynchronously with latest memory list;
           otherwise returns current (possibly out-dated) memory list'''
        if func:
            self.memory_list_func = func
            id = self.remote.YAML_AMS_Comm_get_memory_list(self.comm,self)
            args[id] = ['YAML_AMS_Comm_get_memory_list',self.comm]
            sent += 1
        else:
            return self.memlist

    def memory_attach(self,memory,func = null):
        '''Pass in string name of AMS memory object
           If called with func (not yet done) then first updates comm with latest memory list and then calls func with memory'''
        return self.memories[memory]

    def onRemoteResponse(self, response, request_info):
        global args,sent,recv
        recv += 1
        method = str(request_info.method)
        rid    = request_info.id
        if method == "YAML_AMS_Connect":
            if isinstance(response,list):
#             Currently always connects to the zeroth communicator published, no way to connect to others.
              self.commname = str(response[0])
            else:
              self.commname = str(response)
            if self.commname == 'No AMS publisher running':
                 pass
            else:
                 id = self.remote.YAML_AMS_Comm_attach(self.commname,self)
                 args[id] = ['YAML_AMS_Comm_attach',self.commname]
                 sent += 1
        elif method == "YAML_AMS_Comm_attach":
            self.comm = str(response)
            id = self.remote.YAML_AMS_Comm_get_memory_list(self.comm,self)
            args[id] = ['YAML_AMS_Comm_get_memory_list',self.comm]
            sent += 1
        elif method == "YAML_AMS_Comm_get_memory_list":
            if not isinstance(response,list):response = [response]
            self.memlist = response
            for i in self.memlist:
                self.memories[i] = AMS_Memory(self.comm,i)
            if self.memory_list_func:
                self.memory_list_func(response)
                self.memory_list_func = null

    def onRemoteError(self, code, errobj, request_info):
        global statusbar,recv
        recv += 1
        statusbar.setText("Error "+str(errobj))

if __name__ == '__main__':
    pyjd.setup()
    app = AMSJavascriptExample()
    app.onModuleLoad()
    pyjd.run()

