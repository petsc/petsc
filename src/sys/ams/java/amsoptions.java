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

/*
    This is the class that this file implements (must always be the same as
  the filename.
*/
public class amsoptions {
    
  /* Entire GUI window */
  JFrame frame;

  /*  Portion of GUI window that excludes continue and quit buttons */
  JPanel panel;
  JPanel tpanel;
    
  /*  AMSBean Object - this is where all the AMS "global" functions and 
                       "enum" types are stored                            */
  AMSBean amsbean;

  /* Current PETSc communicator and memory; number of option sequences set */
  String     petsccomm;
  AMS_Comm   ams;
  AMS_Memory mem;
  int        count = 0;

  /*
     The constructor for the amsoptions class. Builds the object (AMSBean)
     for using the AMS
  */
  public amsoptions() {/*----------------------------------------------------------*/
    amsbean = new AMSBean() {
      public void print_error(String mess){  /* overwrite the error message output*/
        System.out.println("AMS Error Message: "+mess);
        /* throw new RuntimeException("Stack traceback"); */
      }
    };
  }

  public void createGUI() {/*-----------------------------------------------------*/

    /* Create main window */
    frame = new JFrame("PETSc Options Setter");
    frame.addWindowListener(new WindowAdapter() {
      public void windowClosing(WindowEvent e) {
        System.exit(0);
      }
    });
        
    tpanel = new JPanel(new GridLayout(1,1));
    frame.getContentPane().add(tpanel, BorderLayout.NORTH);

    /* Create text area where choices will be displayed */
    panel = new JPanel(new GridLayout(0,2));
        
    /* Put the text area in a scroller. */
    JScrollPane scroll = new JScrollPane();
    scroll.getViewport().add(panel);
    	
    frame.getContentPane().add(scroll,BorderLayout.CENTER);

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
    frame.getContentPane().add(bpanel, BorderLayout.SOUTH);
        
    frame.pack();
    frame.setVisible(true);
  }    
    
  public void start(String args[]) { /* ------------------------------------------*/
    createGUI();

    /* Process -ams_server and -ams_port command line options */
    int    i, port = -1;
    String host = "localhost";

    for (i=0; i<args.length; i++) {
      if (args[i].equals("-ams_server")) {
        if (i == args.length-1) {
          System.out.println("Need server name after the -ams_server option");
          System.exit(1);
        }
        host = args[i+1];
     }
     if (args[i].equals("-ams_port")) {
       if (i == args.length-1) {
         System.out.println("Need port number after the -ams_port option");
         System.exit(1);
       }
       port = Integer.parseInt(args[i+1]);
    }
  }

  /* Get list of communicators */
  String list[] = AMSBean.get_comm_list(host,port);
  if (list == null) {
    System.out.println("Unable to connect to publisher");
    System.exit(1);
  }

  /* look for PETSc communicators */
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
      System.exit(1);
    }
    petsccomm = list[i];          

    /*    AMSBean.set_output_file("/dev/null");  */

    /* Attach to the PETSc Communicator */
    ams = AMSBean.get_comm(petsccomm);

    if (ams == null) {
      System.out.println("Could not get communicator:"+petsccomm);
      System.exit(1);
    }
    displayoptionsset();
  }
    
  /*

  */
  public void displayoptionsset() { /*---------------------------------------------*/
    /*
            Clear the window of old options
    */
    panel.removeAll();
    tpanel.removeAll();
        
    String options = "Options_"+count++;
    /* Get the options memory (we ignore the rest) */
    mem = ams.get_memory(options);
    while (mem == null) {
      try {Thread.sleep(300);} catch (InterruptedException ex) {;} finally {;} 
      mem = ams.get_memory(options);
    }

    /* first field is always the name of the options being set */
    String flist[] = mem.get_field_list();
    String OptionsCategory = flist[0];
    String Prefix = mem.get_field(flist[0]).getStringData()[0];
    if (Prefix != null) {
      if (Prefix.equals("Exit")) {
	  /* Tell PETSc program we have received exit message */
	  System.out.println("Received exit from program");
	  mem.unlock();
	  System.exit(0);
      }
      OptionsCategory = Prefix+":"+OptionsCategory;
    }
    Label  label = new Label(OptionsCategory);
    label.setAlignment(Label.CENTER);
    label.setForeground(Color.red);
    tpanel.add(label);

    boolean left = false;
    /* Loop over the rest of the fields */
    int i;
    for (i=2; i<flist.length; i++) {
      AMS_Field lockfld    = mem.get_field(flist[i]);
      String    man[]      = mem.get_field(flist[i+1]).getStringData();
      AMS_Field fld        = mem.get_field(flist[i+2]);
      int       info[]     = fld.get_field_info();
      String    tag        = flist[i+2]+" ("+flist[i]+")";
      left = !left;

      /* handle OptionsSelectInt() */
      if (info[1] == AMSBean.INT) {
        int value[] = fld.getIntData();
        panel.add(new JPanelPack(new TextFieldInt(flist[i],flist[i+2],value[0]),new Label(tag)));

      } else if (info[1] == AMSBean.BOOLEAN) {

        /* is it a header label? */
        if (flist[i].substring(0,8).equals("-amshead")) {
          if (!left) {
            panel.add(new Label(" "));
          }
          int len = flist[i+2].length();
          label = new Label(flist[i+2].substring(0,1+len/2));
          label.setAlignment(Label.RIGHT);
          label.setForeground(Color.red);
          panel.add(label);
          label = new Label(flist[i+2].substring(1+len/2));
          label.setAlignment(Label.LEFT);
          label.setForeground(Color.red);
          panel.add(label);
        } else {
          boolean value[] = fld.getBooleanData();
          MyCheckbox checkbox;
          checkbox = new MyCheckbox(flist[i],flist[i+2],value[0]);
          checkbox.addItemListener(new MyCheckboxItemListener());
          panel.add(checkbox);
        }

      /* handle OptionsSelectDouble() */
      } else if (info[1] == AMSBean.DOUBLE) {
        double value[] = fld.getDoubleData();
        panel.add(new JPanelPack(new TextFieldDouble(flist[i],flist[i+2],value[0]),new Label(tag)));

      /* handle string */
      } else if (info[1] == AMSBean.STRING) {
        String value[] = fld.getStringData();
       
        /* handle OptionsSelectList() */
        if (flist[i+2].length() > 8 && (flist[i+2].substring(0,8)).equals("DEFAULT:")) {
          int       j;
          AMS_Field lfld    = mem.get_field(flist[i+3]);
          String    llist[] = lfld.getStringData();
          MyChoice  choice = new MyChoice(flist[i],flist[i+2]);

          choice.addItem(value[0]);
          for (j=0; j<llist.length; j++) {
            if (!llist[j].equals(value[0])) {
              choice.addItem(llist[j]);
            }
          }
          choice.addItemListener(new MyChoiceItemListener());
          panel.add(new JPanelPack(choice,new Label(flist[i+2].substring(8)+" ("+flist[i]+")")));
          i++;

        /* handle OptionsSelectString() */
        } else {
          panel.add(new JPanelPack(new TextFieldString(flist[i],flist[i+2],value[0]),new Label(tag)));
        }
      }
      i++; i++;
    }
    frame.pack();
    frame.show();
  }
    
    
  /*
      These are INNER classes; they provide callbacks to the window buttons
  */
  /* callback for the quit button */
  class QuitActionListener implements ActionListener {/*------------------------*/
    public void actionPerformed(ActionEvent e) {
      System.out.println("User selected quit");
      System.exit(1);
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
      System.out.println("User changed checkbox"+checkbox.getLabel());
      mem.get_field(checkbox.vName).setData(checkbox.getState(),0);
      mem.get_field(checkbox.vLock).setData(true,0);
    }
  }

  class MyCheckbox extends Checkbox { /*----------------------------------------*/
    String vLock,vName;
    public MyCheckbox(String vlock,String vname,boolean v) {
      super(vname+" ("+vlock+")",v);
      vLock = vlock;
      vName = vname;
    }
  }

  /* call back for the select option */
  class MyChoiceItemListener implements ItemListener {/*--------------------*/
    public void itemStateChanged(ItemEvent e) {
      MyChoice choice = (MyChoice) e.getItemSelectable();
      String   oldata = mem.get_field(choice.vName).getStringData()[0];

      if (!oldata.equals(choice.getSelectedItem())) {
        mem.get_field(choice.vName).setData(choice.getSelectedItem(),0);
        System.out.println("User changed Choice"+choice.getSelectedItem()+mem.get_field(choice.vName).getStringData()[0]);
        mem.get_field(choice.vLock).setData(true,0);

        /* tell publisher that I changed a method so it can send me a new screen of data */
        mem.get_field("ChangedMethod").setData(true,0);
        panel.removeAll();
        System.out.println("User selected choice");
        (new ThreadOptionUpdate()).start();
      }
    }
  }

  class MyChoice extends Choice { /*----------------------------------------*/
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
      super(12); /* create text field with 12 columns */
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
      super(12);
      setValue(value);
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
    }
  }

    public class JPanelPack extends JPanel { /*-----------------------------------*/
      public JPanelPack(Component c1,Component c2) {
        super( new GridBagLayout());
        add(c1);
        add(c2);
        GridBagLayout layout = (GridBagLayout) getLayout();
        GridBagConstraints constraints = new GridBagConstraints();
        constraints.anchor  = GridBagConstraints.WEST;
        constraints.weightx = 100;
        constraints.gridx   = GridBagConstraints.RELATIVE;
        layout.setConstraints(c2,constraints);
      }
    }

  /*
      The main() program. Creates an amsoptions object and calls the start()
    method on it.
  */

  public static void main(String s[]) {  /*-------------------------------------*/
    amsoptions amsoptions = new amsoptions();
    amsoptions.start(s);
  }

}




