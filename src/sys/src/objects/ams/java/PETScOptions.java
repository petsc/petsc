/*$Id: PETScOptions.java,v 1.6 2001/02/14 19:38:54 bsmith Exp bsmith $*/
/*
     Accesses the PETSc published database options and allows the user to change them via a GUI
*/

/*  These are the Java GUI classes */
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

/* For the text input regions */
import javax.swing.text.*;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.Locale;

/*   This allows multiple threads */
import java.lang.Thread;

/*  These are the AMS API classes */
import gov.anl.mcs.ams.*;

import java.net.*;
import java.awt.print.*;
import java.security.*;

/*
    This is the class that this file implements (must always be the same as
  the filename).

    Applet is a subclass of PanelFrame, i.e. it is itself the base window we draw onto
*/
public class PETScOptions extends JApplet {

  boolean checking;

  /*  top panel that shows options title */
  JPanel tpanel;
  /*  middle panel shows selections */
  JPanel panel;
    
  /*  AMSBean Object - this is where all the AMS "global" functions and 
                       "enum" types are stored                            */
  AMSBean amsbean;

  /* Current PETSc communicator and memory; number of option sequences set */
  String     petsccomm;
  AMS_Comm   ams;
  AMS_Memory mem;
  int        count = 0;

  String     host = "terra.mcs.anl.gov";
  int        port = 9000;

  boolean    waiting = false; /* indicates choices have been presented on screen, waiting for user input */

  java.applet.AppletContext appletcontext;
  PETScOptions              applet;
  Container                 japplet;

  JTextField inputport;
  JTextField inputserver;

  /*
    This is the destructor;
  */
  public void destroy(){
        System.out.println("destroy called");
  }

  public void init(){
    applet = this;
    System.out.println("PETScOptions: codebase:"+this.getDocumentBase()+":");
    System.out.println("PETScOptions: about to load amsacc library ");

    try {
      amsbean = new AMSBean() {
        public void print_error(String mess) {  /* overwrite the error message output*/
          System.out.println("AMS Error Message: "+mess);
        }
      };
    } catch (UnsatisfiedLinkError oops) {
      try {
        this.getAppletContext().showDocument(new URL("http://www.mcs.anl.gov/petsc/plugins-amsacc.html"));
      } catch (java.net.MalformedURLException ex) {;}
    } catch (AccessControlException oops) {
      try {
        this.getAppletContext().showDocument(new URL("http://www.mcs.anl.gov/petsc/plugins-security.html"));
      } catch (java.net.MalformedURLException ex) {;}
    } catch (ExceptionInInitializerError oops) {
      try {
        this.getAppletContext().showDocument(new URL("http://www.mcs.anl.gov/petsc/plugins-security.html"));
      } catch (java.net.MalformedURLException ex) {;}
    } 
    System.out.println("PETScOptions: done loading amsacc library ");

    appletcontext = this.getAppletContext();
    japplet       = this.getContentPane();
    System.out.println("PETScOptions: done creating AMSBean ");
  }
 
  public String getAppletInfo() {
    return "Set PETSc obtions via the AMS";
  }

  public void stop() {
        System.out.println("Called stop");
  }

  /*
       This is called by the applet and is much like a main() program, except that 
    if other threads exist the applet does not end when this routine ends.
  */
  public void start() { /* ------------------------------------------*/
     getserver();
  }
    
  public class JPanelSimplePack extends JPanel { /*-----------------------------------*/
    public JPanelSimplePack(String text,Component c1) {
      super( new GridBagLayout());
      add(new JLabel(text));
      add(c1);
    }
  }

  public void getserver() { /* ------------------------------------------*/
    checking = false;
    japplet.removeAll();
    japplet.setVisible(false);
    /*

         Make GUI to get host and port number from user 
    */
    japplet.setLayout(new FlowLayout());
        
    tpanel = new JPanel(new GridLayout(3,1));
    japplet.add(tpanel, BorderLayout.NORTH);
        
    inputserver = new JTextField(host,32);
    JPanelSimplePack text = new JPanelSimplePack("AMS Client machine",inputserver);
    tpanel.add(text);
    inputport = new JTextField(port+"",8);
    text = new JPanelSimplePack("AMS Client port",inputport);
    tpanel.add(text);
    System.out.println("put up server and port");
    
    /*--------------------- */
    JButton button = new JButton("Continue");
    tpanel.add(button);
    button.addActionListener(new ActionListener(){
      public void actionPerformed(ActionEvent e) { 
        System.out.println("User selected continue");
        connect();
      }
    }); 
    System.out.println("put up continue");

    japplet.setVisible(true);
    japplet.validate(); 
    japplet.repaint(); 
    System.out.println("put up continue done");
    return;
  }

  /*
      Connect to the PETSc program and display the first set of options
  */
  public void connect() { /* ------------------------------------------*/

    System.out.println("in connect");

    host = inputserver.getText();
    port = (new Integer(inputport.getText())).intValue();
        
    japplet.removeAll();
    japplet.setVisible(false);

    /* Get list of communicators */
    String list[] = AMSBean.get_comm_list(host,port);
    if (list == null) {
      System.out.println("Unable to connect to publisher on "+host+" "+port);
      getserver();
      return;
    }

    /* look for PETSc communicators */
    int i;
    for (i=0; i<list.length; i++) {
      if ((list[i].substring(0,5)).equals("PETSc")) {
        break;
      }
    }
        
    if (i == list.length) {
      System.out.println("Publisher does not have PETSc communicator. Communicator has");
        for (i=0; i<list.length; i++) {
          System.out.println(list[i]);
        }
      getserver();
      return;
    }
    petsccomm = list[i];          

    /* Attach to the PETSc Communicator */
    ams = AMSBean.get_comm(petsccomm);

    if (ams == null) {
      System.out.println("Could not get communicator:"+petsccomm);
      getserver();
      return;
    }
    displayoptionsset();
  }
    
  /*
        Displays the current set of options and sets up call backs for all options
  */
  public void displayoptionsset() { /*---------------------------------------------*/
    /*
            Clear the window of old options
    */
    System.out.println("About to remove panels");    
    japplet.removeAll();
    japplet.setVisible(false);
    System.out.println("Removed panel; trying to get options");    

    JButton done = new JButton("Done");
    japplet.add(done);
    done.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        japplet.removeAll();
        japplet.setVisible(false);
        System.out.println("User selected done");
        getserver();
      }
    });
    japplet.setVisible(true);
    japplet.validate(); 
    japplet.repaint(); 
        
    String options = "Options_"+count++;
    /* Get the options memory (we ignore the rest) */
    mem = ams.get_memory(options);
    checking = true;
    while (mem == null) {
      try {Thread.sleep(300);} catch (InterruptedException ex) {;} finally {;} 
      if (!checking) return;
      mem = ams.get_memory(options);
    }

    System.out.println("Got next set of options");    
    japplet.removeAll();
    japplet.setVisible(false);

    japplet.setLayout(new FlowLayout());
        
    tpanel = new JPanel(new GridLayout(1,1));
    japplet.add(tpanel, BorderLayout.NORTH);

    /* Create text area where choices will be displayed */
    panel = new JPanel(new GridLayout(0,2));
        
    /* Put the text area in a scroller. */
    JScrollPane scroll = new JScrollPane();
    scroll.getViewport().add(panel);
    japplet.add(scroll,BorderLayout.CENTER);

    /* Create a Panel that  will contain two buttons. Default layout manager */
    JPanel bpanel = new JPanel(new GridLayout(1,2));
        
    /* Add two buttons */
    JButton button = new JButton("Continue");
    bpanel.add(button);
    button.addActionListener(new ContinueActionListener()); /* callback for button */
        
    button = new JButton("Quit");
    bpanel.add(button);
    button.addActionListener(new QuitActionListener());
        
    /* Add the Panel in the bottom of the Frame */
    japplet.add(bpanel, BorderLayout.SOUTH);


    /* first field is always the name of the options being set */
    String flist[] = mem.get_field_list();
    String OptionsCategory = flist[0];
    String Prefix = mem.get_field(flist[0]).getStringData()[0];
    System.out.println("prefix"+Prefix);    
    if (Prefix != null) {
      OptionsCategory = Prefix+":"+OptionsCategory;
    }
    JLabel  label = new JLabel(OptionsCategory);
    /*label.setAlignment(JLabel.CENTER); */
    label.setForeground(Color.red);
    tpanel.add(label);

    /* second field is mansec */
    String mansec = flist[1];
    System.out.println("mansec"+mansec);    

    boolean left = false;
    /* Loop over the rest of the fields */
    int i;
    for (i=3; i<flist.length; i++) {
      AMS_Field lockfld    = mem.get_field(flist[i]);
      String    man        = mem.get_field(flist[i+1]).getStringData()[0];
      AMS_Field fld        = mem.get_field(flist[i+2]);
      int       info[]     = fld.get_field_info();
      String    tag        = flist[i+2];
      String    tip        = flist[i];
      left = !left;

      /* handle OptionsSelectInt() */
      if (info[1] == AMSBean.INT) {
        int value[] = fld.getIntData();
        panel.add(new JPanelPack(new TextFieldInt(flist[i],flist[i+2],value[0]),tag,tip,man,mansec));

      } else if (info[1] == AMSBean.BOOLEAN) {

        /* is it a header label? If so, print it centered accross the screen */
        if (flist[i].substring(0,8).equals("-amshead")) {
          if (!left) {
            panel.add(new JLabel(" "));
          }
          int len = flist[i+2].length();
          label = new JLabel(flist[i+2].substring(0,1+len/2));
          label.setHorizontalAlignment(SwingConstants.RIGHT); 
          label.setForeground(Color.red);
          panel.add(label);
          label = new JLabel(flist[i+2].substring(1+len/2));
          label.setHorizontalAlignment(SwingConstants.LEFT); 
          label.setForeground(Color.red);
          panel.add(label);
        } else {
          boolean value[] = fld.getBooleanData();
          MyCheckbox checkbox;
          checkbox = new MyCheckbox(flist[i],flist[i+2],value[0]);
          checkbox.addItemListener(new MyCheckboxItemListener());
          panel.add(new JPanelPack(checkbox,tag,tip,man,mansec));
        }

      /* handle OptionsSelectDouble() */
      } else if (info[1] == AMSBean.DOUBLE) {
        double value[] = fld.getDoubleData();
        panel.add(new JPanelPack(new TextFieldDouble(flist[i],flist[i+2],value[0]),tag,tip,man,mansec));

      /* handle string */
      } else if (info[1] == AMSBean.STRING) {
        String value[] = fld.getStringData();
       
        /* handle OptionsSelectList() */
        if ((flist[i+2].substring(0,8)).equals("DEFAULT:")) {
          int       j;
          AMS_Field lfld    = mem.get_field(flist[i+3]);
          String    llist[] = lfld.getStringData();
          MyChoice  choice = new MyChoice(flist[i],flist[i+2]);

    System.out.println("flist[i]"+flist[i]+"flist[i+1]"+flist[i+1]+"flist[i+3]"+flist[i+3]);    

          choice.addItem(value[0]);
          for (j=0; j<llist.length-1; j++) {
            if (!llist[j].equals(value[0])) {
              choice.addItem(llist[j]);
            }
          }
          choice.addItemListener(new MyChoiceItemListener());
          tag = flist[i+2].substring(8);
          panel.add(new JPanelPack(choice,tag,tip,man,mansec));
          i++;

        /* handle OptionsSelectString() */
        } else {
          panel.add(new JPanelPack(new TextFieldString(flist[i],flist[i+2],value[0]),tag,tip,man,mansec));
        }
      }
      i++; i++;
    }
    System.out.println("Processed options set");    
    japplet.setVisible(true);
    japplet.validate(); 
    japplet.repaint(); 
 }
    
    
  /*
      These are INNER classes; they provide callbacks to the window buttons, textedits, etc
  */
  /* callback for the quit button */
  class QuitActionListener implements ActionListener {/*------------------------*/
    public void actionPerformed(ActionEvent e) {
      System.out.println("User selected quit");
      getserver();
    }
  }

  /* callback for the continue button */
  class ContinueActionListener implements ActionListener {/*--------------------*/
    public void actionPerformed(ActionEvent e) {
       panel.removeAll();
      System.out.println("User selected continue");
      (new ThreadOptionUpdate()).start();
    }
  }

  /* call back for the check box */
  class MyCheckboxItemListener implements ItemListener {/*--------------------*/
    public void itemStateChanged(ItemEvent e) {
      MyCheckbox checkbox = (MyCheckbox) e.getItemSelectable();
      System.out.println("User changed checkbox"+checkbox.getText());
      mem.get_field(checkbox.vName).setData(checkbox.isSelected(),0);
      mem.get_field(checkbox.vLock).setData(true,0);
    }
  }

  class MyCheckbox extends JCheckBox { /*----------------------------------------*/
    String vLock,vName;
    public MyCheckbox(String vlock,String vname,boolean v) {
      super("",v);
      vLock = vlock;
      vName = vname;
    }
  }

  /* call back for the select option */
  class MyChoiceItemListener implements ItemListener {/*--------------------*/
    public void itemStateChanged(ItemEvent e) {
      MyChoice choice = (MyChoice) e.getItemSelectable();

      if (e.getStateChange() == ItemEvent.SELECTED) {
        System.out.println("User changing Choice "+choice.vName+"changed to"+(String)choice.getSelectedItem());
        mem.get_field(choice.vName).setData((String)choice.getSelectedItem(),0);
        System.out.println("User changed Choice "+choice.vName+"changed to"+(String)choice.getSelectedItem());
        mem.get_field(choice.vLock).setData(true,0);

        /* tell publisher that I changed a method so it can send me a new screen of data */
        mem.get_field("ChangedMethod").setData(true,0);
        panel.removeAll();
        System.out.println("User selected choice");
        (new ThreadOptionUpdate()).start();
      }
    }
  }

  class MyChoice extends JComboBox { /*----------------------------------------*/
    String vLock,vName;
    public MyChoice(String vlock,String vname) {
      super();
      vLock = vlock;
      vName = vname;
    }
  }

  /* callback for the integer field is the insertString() method below */
  public class TextFieldInt extends JTextField { /*------------------------------*/   
    private NumberFormat integerFormatter;
    private String       vLock,vName;
    public TextFieldInt(String vlock,String vname, int value) {
      super(6); /* create text field with 6 columns */
      integerFormatter = NumberFormat.getNumberInstance(Locale.US);
      integerFormatter.setParseIntegerOnly(true);
      setValue(value);
      vLock = vlock;
      vName = vname;
    }
    public int getValue() {
      try { 
        return integerFormatter.parse(getText()).intValue();
      } catch (ParseException e) {;}
      return 0;
    }

    public void setValue(int value) {
      setText(integerFormatter.format(value));
    }

    protected Document createDefaultModel() {
      return new IntDocument();
    }

    protected class IntDocument extends PlainDocument {
      public void insertString(int offs, String str,AttributeSet a) throws BadLocationException {
	char[] source = str.toCharArray();
	char[] result = new char[source.length];
	int j = 0;
	for (int i = 0; i<result.length; i++) {
	  if (Character.isDigit(source[i])) {
	    result[j++] = source[i];
	  } 
	}
	super.insertString(offs, new String(result,0,j),a);
        if (vName != null) {
          System.out.println("User changed int"+vName+vLock+getValue());
	  mem.get_field(vName).setData(getValue(),0); 
	  mem.get_field(vLock).setData(true,0); 
	}
      }
    }
  }

  /* callback for the double field is the insertString() method below */
  public class TextFieldDouble extends JTextField { /*-----------------------------------*/
    private String vLock,vName;
    public TextFieldDouble(String vlock,String vname, double value) {
      super(String.valueOf(value),Math.max(8,String.valueOf(value).length()));
      vLock = vlock;
      vName = vname;
    }

    public double getValue() {
      return Double.parseDouble(getText());
    }

    public void setValue(double value) {
      setText(String.valueOf(value));
    }

    protected Document createDefaultModel() {
      return new DoubleDocument();
    }

    protected class DoubleDocument extends PlainDocument {
      public void insertString(int offs, String str,AttributeSet a) throws BadLocationException {
	char[] source = str.toCharArray();
	char[] result = new char[source.length];
	int j = 0;
	for (int i = 0; i<result.length; i++) {
	  if (Character.isDigit(source[i]) || source[i] == 'E' ||
              source[i] == '+' || source[i] == '-' || source[i] == '.') {
	    result[j++] = source[i];
	  } 
	}
	super.insertString(offs, new String(result,0,j),a);
        if (vName != null) {
          double v;
          try {
            v = getValue();
          } catch (NumberFormatException ex) {
            return;
          } 
          System.out.println("User changed double"+vName+vLock+v);

	  mem.get_field(vName).setData(v,0); 
	  mem.get_field(vLock).setData(true,0); 
	}
      }
    }
  }

  /* callback for the double string is the insertString() method below */
  public class TextFieldString extends JTextField { /*-----------------------------------*/
    private String vLock,vName;
    public TextFieldString(String vlock,String vname,String value) {
      super(12);
      setText(value);
      vLock = vlock;
      vName = vname;
    }

    protected Document createDefaultModel() {
      return new DoubleDocument();
    }

    protected class DoubleDocument extends PlainDocument {
      public void insertString(int offs, String str,AttributeSet a) throws BadLocationException {
	super.insertString(offs,str,a);
        if (vName != null) {
	  mem.get_field(vName).setData(str,0); 
	  mem.get_field(vLock).setData(true,0); 
	}
      }
    }
  }

  /*
     Methods used by the callbacks (inner classes)
  */

  class ThreadOptionUpdate extends Thread {/*-----------------------------------*/
    public void run() {


      displayoptionsupdate(); /* update options on PETSc program */
      displayoptionsset(); /* wait for next set of options from PETSc program */
    }
  }

  public void displayoptionsupdate() { /*---------------------------------------*/
    if (mem != null) {

      /* Send values through AMS to PETSc program  */
      mem.send_begin();
      mem.send_end(); 
        
      /* Tell PETSc program we are done with this set of options*/
      mem.unlock();
      System.out.println("Options updated in program");
    }
  }

    /* puts two components next to each other on the screen */
  public class JPanelPack extends JPanel { /*-----------------------------------*/
    public JPanelPack(Component c1,String text,String tip,String man,String mansec) {
      super( new GridBagLayout());
      add(c1);
      JButton c2 = new JButton(text);
      /* c2.setToolTipText(tip); does not work in applet */
      c2.setBorderPainted(false);
      if (man.compareTo("None") != 0) {
        c2.setActionCommand(mansec+"/"+man+".html");
      } else {
        c2.setActionCommand("None");
      }
      c2.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          System.out.println("User selected manualpage");
          String iman = e.getActionCommand();
          if (iman.compareTo("None") != 0) {
            String[] sites = new String[2];
            sites[0] = "http://www-unix.mcs.anl.gov/petsc/docs/manualpages/";
            sites[1] = "http://www-unix.mcs.anl.gov/tao/docs/manualpages/";
            int i;
            for (i=0; i<2; i++) {
              URL url = null;
              try {
                url = new URL(sites[i]+iman);
              } catch (MalformedURLException ex) {;}

              /* check if the MCS webserver can find the page */
              java.io.InputStreamReader stream = null;
              try {
                stream = new java.io.InputStreamReader(url.openStream());
              } catch (java.io.IOException ex) {continue;}
              char[] errors = new char[1024];
              try {
                stream.read(errors);
              } catch (java.io.IOException ex) {;}
              String serrors = new String(errors);
              System.out.println(serrors); 
              if (serrors.indexOf("File Not Found") == -1) {
                appletcontext.showDocument(url,"ManualPages");
                break;
              }
            }
          }
        }
      }); 
      add(c2);
      GridBagLayout layout = (GridBagLayout) getLayout();
      GridBagConstraints constraints = new GridBagConstraints();
      constraints.anchor  = GridBagConstraints.WEST;
      constraints.weightx = 100;
      constraints.gridx   = GridBagConstraints.RELATIVE;
      layout.setConstraints(c2,constraints);
    }
  }

}




