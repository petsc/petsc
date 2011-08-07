#   
#   Run python pyjsbuild --output . JSONRPCExample.py to generate the needed HTML and Javascript
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

import time

comm   = -1  # Currently attached AMS communicator; only one is supported at a time
args   = {}  # Arguments to each remote call 


class JSONRPCExample:
    def onModuleLoad(self):
        self.TEXT_WAITING = "Waiting for response..."
        self.TEXT_ERROR = "Server Error"
        self.remote_py = ServicePython()
        self.status=Label()
        self.text_area = TextArea()
        self.text_area.setText("Simple text")
        self.text_area.setCharacterWidth(80)
        self.text_area.setVisibleLines(8)
        
        self.button_echo = Button("Send to echo Service", self)
        self.button_ams_memlist = Button("Get AMS memory list", self)
        self.button_useclass = Button("Useclass", self)
        self.button_useclass2 = Button("Useclass2", self)
        self.button_useclass3 = Button("Useclass3", self)

        buttons = HorizontalPanel()
        buttons.add(self.button_echo)
        buttons.add(self.button_ams_memlist)
        buttons.add(self.button_useclass)
        buttons.add(self.button_useclass2)
        buttons.add(self.button_useclass3)
        buttons.setSpacing(8)
        
        info = """<h2>JSON-RPC Example</h2>
        <p>This example demonstrates the calling of AMS server in PETSc with <a href="http://json-rpc.org/">JSON-RPC</a>.
        </p>"""
        
        self.panel = VerticalPanel()
        self.panel.add(HTML(info))
        self.panel.add(buttons)
        self.panel.add(self.status)
        
        RootPanel().add(self.panel)
        self.panel.add(self.text_area)

        self.commobj = AMS_Comm()

    def joe(self,response):
        self.text_area.setText('kate'+str(response))

    def onClick(self, sender):
        self.status.setText(self.TEXT_WAITING)
        text = self.text_area.getText()

        if sender == self.button_echo:
            id = self.remote_py.YAML_echo(text, self)
            args[id] = ['YAML_echo',text]
        elif sender == self.button_ams_memlist:
            id = self.remote_py.YAML_AMS_Connect(text, self)
            args[id] = ['YAML_AMS_Connect',text]
        elif sender == self.button_useclass:
            self.text_area.setText('joe'+str(self.commobj.get_memory_list())+str(self.commobj.commname))
        elif sender == self.button_useclass2:
            result = self.commobj.get_memory_list(func = self.joe)
            self.text_area.setText('old'+str(result))
        elif sender == self.button_useclass3:
            result = self.commobj.get_memory_list()
            self.text_area.setText('1')
            for i in result:
               self.text_area.setText('2')
               memory = self.commobj.memory_attach(i)
               self.text_area.setText('3')
               fields = memory.get_field_list()
               self.text_area.setText('4')
               for j in fields:
                  self.text_area.setText('5'+j)
                  field = memory.get_field_info(j)
                  self.text_area.setText('6')
                  newstatus2=Label()
                  newstatus2.setText(i+j+str(field))
                  self.panel.add(newstatus2)


    def onRemoteResponse(self, response, request_info):
        global comm
        self.status.setText(response)
        method = str(request_info.method)
        rid    = request_info.id
        if method == "YAML_AMS_Connect":
            id = self.remote_py.YAML_AMS_Comm_attach(str(response),self)
            args[id] = ['YAML_AMS_Comm_attach',str(response)]
        elif method == "YAML_AMS_Comm_attach":
            comm = response  # will only work for one comm
            id = self.remote_py.YAML_AMS_Comm_get_memory_list(comm,self)
            args[id] = ['YAML_AMS_Comm_get_memory_list',comm]
        elif method == "YAML_AMS_Comm_get_memory_list":
            memlist = response
            for i in memlist:
                id = self.remote_py.YAML_AMS_Memory_attach(comm,i,self)
                args[id] = ['YAML_AMS_Memory_attach',comm,i]
        elif method == "YAML_AMS_Memory_attach":
            memory = response[0]
            step   = response[1]
            id = self.remote_py.YAML_AMS_Memory_get_field_list(memory,self)
            args[id] = ['YAML_AMS_Memory_get_field_list',memory]
        elif method == "YAML_AMS_Memory_get_field_list":
            localmemory = args[rid][1]
            for i in response:
                id = self.remote_py.YAML_AMS_Memory_get_field_info(localmemory,i,self)
                args[id] = ['YAML_AMS_Memory_get_field_info',localmemory,i]
        elif method == "YAML_AMS_Memory_get_field_info":
            newstatus2=Label()
            newstatus2.setText(str(args[rid])+str(response))
            self.panel.add(newstatus2)




    def onRemoteError(self, code, errobj, request_info):
        # onRemoteError gets the HTTP error code or 0 and
        # errobj is an jsonrpc 2.0 error dict:
        #     {
        #       'code': jsonrpc-error-code (integer) ,
        #       'message': jsonrpc-error-message (string) ,
        #       'data' : extra-error-data
        #     }
        message = errobj['message']
        if code != 0:
            self.status.setText("HTTP error %d: %s" %(code, message))
        else:
            code = errobj['code']
            self.status.setText("JSONRPC Error %s: %s" %(code, message))



class ServicePython(JSONProxy):
    def __init__(self):
        JSONProxy.__init__(self, "No service name", ["YAML_echo", "YAML_AMS_Connect", "YAML_AMS_Comm_attach", "YAML_AMS_Comm_get_memory_list","YAML_AMS_Memory_attach","YAML_AMS_Memory_get_field_list","YAML_AMS_Memory_get_field_info"])

# ---------------------------------------------------------
class AMS_Memory(JSONProxy):
    def __init__(self,comm,memory):
        global args
        self.comm             = comm
        self.memory           = memory
        self.fieldlist        = []
        self.fields           = {}
        self.remote           = ServicePython()
        id = self.remote.YAML_AMS_Memory_attach(comm,memory,self)
        args[id] = ['YAML_AMS_Memory_attach',comm,memory]

    def get_field_list(self,func = null):
        '''If called with func (net yet supported) then calls func asynchronously with latest memory list;
           otherwise returns current (possibly out-dated) memory list'''
        return self.fieldlist

    def get_field_info(self,field):
        '''Pass in string name of AMS field
           If called with func (not yet done) then first updates comm with latest field list and then calls func with field'''
        return self.fields[field]

    def onRemoteResponse(self, response, request_info):
        method = str(request_info.method)
        rid    = request_info.id
        if method == "YAML_AMS_Memory_attach":
            self.memory = response[0]
            id = self.remote.YAML_AMS_Memory_get_field_list(self.memory,self)
            args[id] = ['YAML_AMS_Memory_get_field_list',self.memory]
        elif method == "YAML_AMS_Memory_get_field_list":
            self.fieldlist = response
            for i in self.fieldlist:
                id = self.remote_py.YAML_AMS_Memory_get_field_info(self.memory,i,self)
                args[id] = ['YAML_AMS_Memory_get_field_info',self.memory,i]
        elif method == "YAML_AMS_Memory_get_field_info":
            self.fields[args[rid][2]] = response


    def onRemoteError(self, code, errobj, request_info):
        pass

# ---------------------------------------------------------
class AMS_Comm(JSONProxy):
    def __init__(self):
        global args
        self.comm             = -1
        self.commname         = ''
        self.memlist          = []
        self.memories         = {}
        self.memory_list_func = null
        self.remote           = ServicePython()
        id = self.remote.YAML_AMS_Connect('No argument', self)
        args[id] = ['YAML_AMS_Connect']
  #      time.sleep(1)

    def get_memory_list(self,func = null):
        '''If called with func then calls func asynchronously with latest memory list;
           otherwise returns current (possibly out-dated) memory list'''
        if func: 
            self.memory_list_func = func
            id = self.remote.YAML_AMS_Comm_get_memory_list(self.comm,self)
            args[id] = ['YAML_AMS_Comm_get_memory_list',self.comm]
        else:
            return self.memlist

    def memory_attach(self,memory):
        '''Pass in string name of AMS memory object
           If called with func (not yet done) then first updates comm with latest memory list and then calls func with memory'''
        return self.memories[memory]

    def onRemoteResponse(self, response, request_info):
        global args
        global comm
        method = str(request_info.method)
        rid    = request_info.id
        if method == "YAML_AMS_Connect":
            self.commname = str(response)
            id = self.remote.YAML_AMS_Comm_attach(self.commname,self)
            args[id] = ['YAML_AMS_Comm_attach',self.commname]
        elif method == "YAML_AMS_Comm_attach":
            self.comm = str(response)
            id = self.remote.YAML_AMS_Comm_get_memory_list(self.comm,self)
            args[id] = ['YAML_AMS_Comm_get_memory_list',self.comm]
        elif method == "YAML_AMS_Comm_get_memory_list":
            self.memlist = response
            for i in self.memlist:
                self.memories[i] = AMS_Memory(comm,i)
            if self.memory_list_func:
                self.memory_list_func(response)
                self.memory_list_func = null

    def onRemoteError(self, code, errobj, request_info):
        pass

if __name__ == '__main__':
    # for pyjd, set up a web server and load the HTML from there:
    # this convinces the browser engine that the AJAX will be loaded
    # from the same URI base as the URL, it's all a bit messy...
    pyjd.setup()
    app = JSONRPCExample()
    app.onModuleLoad()
    pyjd.run()

