import numpy as np
import matplotlib.pyplot as plt
import PhyPraKit as ppk


class Peak:
    """
    
    Parameters
    ----------
    isotope : string
        name of the Isotope used to create the spectrum.
    description : string
        Additional Information for the spectrum.
    book_value : int
        Known Value for the energy of the peak
    spectrum : int_array
        histogram measured by detector, length = number of channels
    position : int
        (assumed) channel of the peak 
    p_left : int
        channel number for the left peak boarder
    p_right : int
        channel number for the right peak boarder 
        
        
    """
    def __init__(self, isotope, description, book_value, spectrum, position, p_left, p_right):

        self.isotope = isotope
        self.desc = description
        self.book_value = book_value
        
        self.position = position      
        self.pmin = p_left
        self.pmax = p_right
        
        self.iniNb = ((spectrum[p_left] + spectrum[p_right])/2) * (p_right-p_left)

        self.channel = np.array(range(len(spectrum)))
        self.spectrum = spectrum
        
        
        self.peak_channel = np.arange(p_left,p_right+1) #+1 on right
        self.peak_spectrum = spectrum[p_left:p_right]
        
        self.fit = None
        

    def gauss_back(self, x, mu=300, sig=50., Ns=1000, Nb=100., s=0.0) :
        """
        Description
        ----------       
        Gaussian shape on linear background
        
        Parameter
        ----------
        mu: int
            peak posision
        sig: int
            peak width in sigma
        Ns: int
            number of signal events 
        Nb: int
            mumber of background events in interval [mn, mx]
        s: int
            slope of base line
        self.pmin : int
            lower bound of fit interval (as fixed parameter)
        self.pmax: int
            upper bound of fit interval (as fixed parameter)
        """
        # Gaussian signal 
        S = np.exp(-0.5*((x-mu)/sig)**2) / sig / np.sqrt(2*np.pi)
        
        # linear background model
        B = (1 + (x-mu)*s) / (s/2*(self.pmax**2 - self.pmin**2) + (1-s*mu)*(self.pmax - self.pmin))
        return Ns*S + Nb*B

    
    def do_fit(self, do_plot = False, return_Object = False):
        """
        Does fit based on the hfit function from ppk
     
        """
        self.fit = ppk.hFit(
            self.gauss_back,
            self.peak_spectrum,
            self.peak_channel, 
            p0=[self.position, 20, 1000, self.iniNb, 0.0],  # initial guess for parameter values
            limits = (["sig",0.0,None], ["Ns",0.0,None], ["mu", self.pmin , self.pmax]),  # limits
            use_GaussApprox=False,  # Gaussian approximation
            fit_density=False,  # fit density
            
            plot=do_plot,  # plot data and model
            plot_band=do_plot,  # plot model confidence-
            plot_cor=False,  # plot profiles likelihood and contours
            quiet=True,  # suppress informative printout
            same_plot=False,
                        
            axis_labels=["channel", "N"],
            data_legend=self.isotope,
            model_legend="Gaussian shape on linear background",
            return_fitObject= return_Object       
            )
        
        if do_plot : 
            plt.savefig('peak_pictures/'+ self.isotope + self.desc + 'peak.png', dpi = 400)
            plt.clf()
            plt.close()
        
    def plot_input(self,log=False):
        # Create the plot
       plt.figure(figsize=(10, 6))
       plt.plot(self.channel, self.spectrum, label=self.isotope, marker='o', linestyle='', color='b', markersize=1)

       # Add labels and title
       plt.xlabel('Index')
       plt.ylabel('Intensity (arbitrary units)')
       plt.title('Spectrum Plot')
       plt.grid(True)
       plt.legend()       
       
       if(log): plt.yscale('log')

       # Display the plot
       plt.show()

    def plot_peak(self,log=False):
        # Create the plot
       plt.figure(figsize=(10, 6))
       plt.plot(self.channel, self.spectrum, label=self.isotope, 
                marker='o', linestyle='', color='b', markersize=1)
       plt.plot(self.peak_channel, self.peak_spectrum, label=self.isotope, 
                marker='o', linestyle='', color='r', markersize=1)
       
       # Add labels and title
       plt.xlabel('Index')
       plt.ylabel('Intensity (arbitrary units)')
       plt.title('Spectrum Plot')
       plt.grid(True)
       plt.legend()      
       
       if(log): plt.yscale('log')
       
       # Display the plot
       plt.show()



















