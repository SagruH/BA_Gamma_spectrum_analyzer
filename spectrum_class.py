import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks, peak_widths

import yaml
import pandas as pd
import json
import os

from peak_class import Peak

class Spectrum:
    """

    Parameters
    ----------
    isotope : string
        name of the Isotope used to create the spectrum.
    description : string
        Additional Information for the spectrum.
    book_value : array of float
        Bookvalues in keV for each peak, length equal to number of peaks
    tool : string
        information on the device used for measuring 
        currently either "Radiacode" or "Lab_old" pr "Lab_new"
    file : string
        path to file including filename containing data.


    Optional Parameters
    -------
    ini_peak_finders : array of length 5
        initial values für peak finder: prominence, distance, height, threshold, width
        
    restriced_peaks : array of length 2 [Bool,Array]
        if Bool is True only peak numbers from array are used 

    """
    def __init__(self, isotope, description, book_value, tool, file, ini_peak_finder=[1,10,10,5,0.95], restricted_peaks = [False,[0]]):

        #constant attributes
        self.isotope = isotope
        self.desc = description
        self.path = file
        self.tool = tool
        self.save_path = 'peak_data/' + self.isotope + self.desc + '_pdata.json' #path to peak data
        self.book_value = book_value
        
        #read data and write spectrum in spectrum
        if tool == "Radiacode":
            data = self.read_yaml_file(file)
            self.spectrum = np.array(data['spectrum'])
            
        if tool == "Lab_old":
            data = pd.read_csv(file, delimiter=';', skiprows=2, decimal =',')
            self.spectrum = data.iloc[:, 1].to_numpy()
        
        if tool == "Lab_new":
            self.spectrum = np.loadtxt(file, skiprows=0, dtype=int) 
            reshaped_spectrum = self.spectrum.reshape(1024,4)
            self.spectrum = reshaped_spectrum.sum(axis=1)
            
        
        #more attributes 
        self.channel = np.array(range(len(self.spectrum)))
        
        #peak and peak finding attributes
        self.peak_pos = None
        self.peak_prop = None
        self.peak_widths = None
        self.ini_pf = ini_peak_finder
        self.peaks = []
        
        # check for peak restriction
        if restricted_peaks[0]:
            self.peak_restriction = True
            self.restriced_peaks = restricted_peaks[1]
        else: 
            self.peak_restriction = False

            
        #check if peak data is already found and saved
        try:
            self.peak_data = pd.read_csv(self.save_path, sep=';')
            self.peaks_found = True
        except FileNotFoundError:
            self.peaks_found = False

#--------------------------------------------------------------------------
#Misc Functions
            
    def read_yaml_file(self, file_path):    #reads yaml files from radiacode
        try:
            with open(file_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
                return yaml_data

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
    
    
    def load_peak_data(self):
    # Load the data from the JSON file
        with open(self.save_path, 'r') as file:
            data = json.load(file)
        
        # Extract individual variables
        self.ini_pf = [data['ini_prominence'],
                       data['ini_distance'],
                       data['ini_height'],
                       data['ini_threshold'],
                       data['ini_width']]

        self.peak_pos = np.array(data['peak_pos'])
        self.peak_widths = np.array(data['peak_widths'])
        
        # #Debug:
        # print(self.ini_pf)
        # print(self.peak_pos)
        # print(self.peak_widths)
        
#--------------------------------------------------------------------------
#Peak Finding Function
    
    def peak_finder(self):
        """
        
        Usage
        ----------
        Creates a Plot with sliders to adjust the parameters of peak_finder from scipy
        
        Contains two innerfunctions updata and save.
            Update: Allows the Slider to work
            Save: Creates a txt file containg information about the found peaks as well as the used parameters.
        
        """
        
        def update(val):
            prominence = slider_prominence.val
            distance = int(slider_distance.val)
            height = slider_height.val
            threshold = slider_threshold.val
            
            width = slider_width.val
            
            # Update peak detection
            peaks, properties = find_peaks(self.spectrum, prominence=prominence, distance=distance, height=height, threshold=threshold)
            
            # Update the plot with new peaks
            peak_line.set_data(self.channel[peaks], self.spectrum[peaks])
            
            # Update the peak widths
            widths, width_heights, left_ips, right_ips = peak_widths(self.spectrum, peaks, rel_height=width)
            width_line.set_segments(
                [[[self.channel[l], width_heights[i]], [self.channel[r], width_heights[i]]] for i, (l, r) in enumerate(zip(left_ips.astype(int), right_ips.astype(int)))])
            
            #save new found peaks
            self.peak_pos = peaks
            self.peak_prop = properties
            self.peak_widths = widths
            
            fig.canvas.draw_idle()
        
        def save(event):
            data = {
                'ini_prominence': slider_prominence.val,
                'ini_distance': slider_distance.val,
                'ini_height': slider_height.val,
                'ini_threshold': slider_threshold.val,
                'ini_width': slider_width.val,
                'peak_pos': self.peak_pos.tolist(),
                'peak_widths': self.peak_widths.tolist(),
            }
            
            # Ensure the folder exists
            folder_name = "peak_data"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            # #Debug:
            # print(self.peak_pos)
            # print(self.peak_prop)
            # print(self.peak_widths)
            
            # Save the current slider values to a text file
            with open(self.save_path, 'w') as file:
                json.dump(data, file, indent=4)
            print("Slider and Peak values saved")
        
        # Create the initial plot
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.35)
        line, = ax.plot(self.channel, self.spectrum, label= self.isotope +' Spectrum', 
                        marker='o', linestyle='', color='b', markersize=1)
        
        # find intial peaks
        peaks, properties = find_peaks(self.spectrum, prominence=self.ini_pf[0], distance=self.ini_pf[1], 
          
                                       height=self.ini_pf[2], threshold=self.ini_pf[3])
        
        # Plot detected peaks initially
        peak_line, = ax.plot(self.channel[peaks], self.spectrum[peaks], 
                             "x", label='Detected Peaks', color='red', markersize=5)
        ax.set_title('Peak Detection in ' + self.isotope + ' Spectrum')
        ax.set_xlabel('Channel')
        ax.set_ylabel('N')
        ax.legend()
        
        
        # Calculate and plot inital peak widths
        widths, width_heights, left_ips, right_ips = peak_widths(self.spectrum, peaks, rel_height=self.ini_pf[4])
        width_line = ax.hlines(width_heights, self.channel[left_ips.astype(int)], 
                               self.channel[right_ips.astype(int)],
                               color="green", label="Peak Widths")


        # Slider setup
        axcolor = 'lightgoldenrodyellow'
        ax_prominence = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
        ax_distance = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
        ax_height = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        ax_threshold = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
        
        ax_width = plt.axes([0.25, 0.05, 0.50, 0.03], facecolor=axcolor)

        slider_prominence = Slider(ax_prominence, 'Prominence', 0.01, 100.0, valinit=self.ini_pf[0])
        slider_distance = Slider(ax_distance, 'Distance', 1, 1000, valinit=self.ini_pf[1])
        slider_height = Slider(ax_height, 'Height', 0, 1000, valinit=self.ini_pf[2])
        slider_threshold = Slider(ax_threshold, 'Threshold', 0.0, 100.0, valinit=self.ini_pf[3])
        
        slider_width = Slider(ax_width, 'Width', 0.0, 1.0, valinit=self.ini_pf[4])
        
        # Button setup
        ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button = Button(ax_button, 'Save', color=axcolor, hovercolor='0.975')   
        
        # Connect sliders to update function
        slider_prominence.on_changed(update)
        slider_distance.on_changed(update)
        slider_height.on_changed(update)
        slider_threshold.on_changed(update)
        
        slider_width.on_changed(update)

        # Connect Button to save function
        self.button.on_clicked(save)
        
#--------------------------------------------------------------------------
#Plotting Functions
    
    def quick_plot(self,plot_peaks = False, log = False, save = False):
        # find xlim
        xlim = len(self.channel)
       # Create base plot
        plt.hist(self.channel, bins=len(self.channel), weights=self.spectrum, 
                 histtype='step', color='black', label=self.isotope + ' Gamma Spectrum', linewidth=1.5)
        
        # If peaks are to be plotted, highlight them with red markers
        if plot_peaks:
            for i in self.peak_pos:
                plt.plot(self.channel[i], self.spectrum[i],
                         marker='x', linestyle='', color='red', markersize=6)  # No black for simulation
   
    
        # Add labels, title and make pretty
        plt.xlim(0, xlim)
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('Entries', fontsize=12)
        #plt.title('Spectrum Plot' + " " + self.isotope + " " + self.desc)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if(log): plt.yscale('log')
        
        if save:
            plt.savefig('quick_pic/'+ self.isotope + self.desc + 'quickpic.png', dpi = 400)
        # Display the plot
        plt.show()

       
    def full_plot(self,save = False):
        plt.figure(figsize=(7, 4))
        
        # Plot the spectrum as a histogram
        plt.bar(self.channel, self.spectrum, color='gray', label='Spectrum', width=1, alpha=0.7)
        
        # Colors for the peaks and confidence bands
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.peaks))) 
        
        # Iterate over each peak
        for i in range(len(self.peaks)):
            
            # Extract peak fit parameters
            peak_fit = self.peaks[i].fit['parameter values']  
            confidence_intervals = self.peaks[i].fit['confidence intervals']  
            
            #Adjust peak size
            margin = 0.2
            x_min = int(self.peak_pos[i] - self.peak_widths[i] * (1 + margin) / 2)
            x_max = int(self.peak_pos[i] + self.peak_widths[i] * (1 + margin) / 2)
            
            # Slice the spectrum to the relevant x-axis range
            peak_range = np.arange(x_min, x_max + 1)
                    
            # Calculate the lower and upper bounds for the confidence intervals
            lower_params = peak_fit + confidence_intervals[:, 0]
            upper_params = peak_fit + confidence_intervals[:, 1]
            
            # Plot the peak
            peak_curve = self.peaks[i].gauss_back(peak_range, *peak_fit)
            plt.plot(peak_range, peak_curve, color=colors[i], label=f'Peak {i+1}')
            
            # Plot the confidence interval band
            lower_bound = self.peaks[i].gauss_back(peak_range, *lower_params)
            upper_bound = self.peaks[i].gauss_back(peak_range, *upper_params)
            plt.fill_between(peak_range, lower_bound, upper_bound, color=colors[i], alpha=0.2, label=f'Confidence Interval Peak {i+1}')
        
        # Customize the plot
        #plt.title('Spectrum with Fitted Peaks and Confidence Intervals')
        plt.xlabel('Kanal')
        plt.ylabel('N')
        plt.legend(loc='best', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.grid(True)
        
        # Show the plot
        plt.tight_layout()
        if(save): 
            plt.savefig('peak_pictures/'+ self.isotope + self.desc + '_fullpic.png', dpi = 400)
            plt.clf()
            plt.close()
            print(f"Picture saved to {'peak_pictures/'+ self.isotope + self.desc + '_fullpic.png'}")
            return
        plt.show()
       
        
        
#--------------------------------------------------------------------------
#Save Function       
               
        
    def save_spectrum_results(self):
        
        # Ensure the folder exists
        folder_name = "spectrum_results"
        file_name = self.isotope + self.desc + '_spectrum_data.json' #path to peak data    
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        # Build the path for the file
        file_path = os.path.join(folder_name, file_name)
        
        #extract fit_data from peaks
        
        fit_data_array = []
        for i in self.peaks:
            fit_data_array.append(i.fit)   
    
        # Initialize the formatted data container
        formatted_data = {
            "isotope": self.isotope ,
            "description": self.desc ,
            "tool": self.tool ,
            "peaks": []
        }
    
        # Loop over each fit data entry and its corresponding book_mu value
        for peak_index, fit_data in enumerate(fit_data_array):
            parameter_values = fit_data.get('parameter values')
            confidence_intervals = fit_data.get('confidence intervals')
            parameter_names = fit_data.get('parameter names')
            goodness_of_fit = fit_data.get('goodness-of-fit')
            
            peak_data = {
                "goodness_of_fit": goodness_of_fit,
                "parameters": []
            }
    
            # Process each parameter for the current peak
            for i, param_name in enumerate(parameter_names):
                value = parameter_values[i]
                lower_bound, upper_bound = confidence_intervals[i]
                error = max(abs(lower_bound), abs(upper_bound))  # Taking the max error from confidence interval
                
                # For 'mu', add book value comparison
                if param_name == 'mu':
                    peak_data["parameters"].append({
                        "name": param_name,
                        "value": value,
                        "error": error,
                        "book_value": self.book_value[peak_index]  # Add the corresponding book value for this peak in keV          
                    })
                    
                elif param_name == 'sig':
                    peak_data["parameters"].append({
                        "name": param_name,
                        "value": value,
                        "error": error,
                        "fwhm": 2*np.sqrt(2*np.log(2)) * value, #FWHM based on a cont. factor times sigma
                        "fwhm_error": 2*np.sqrt(2*np.log(2)) * error
                    })
                else:
                    # Append other parameter data
                    peak_data["parameters"].append({
                        "name": param_name,
                        "value": value,
                        "error": error
                    })
            
            # Append the current peak's data to the main list
            formatted_data["peaks"].append(peak_data)
    
        # Build the path for the file
        file_path = os.path.join(folder_name, file_name)
    
        # Save the data to the file as JSON
        with open(file_path, 'w') as json_file:
            json.dump(formatted_data, json_file, indent=4)
    
        print(f"Data saved to {file_path}")
       
       
#--------------------------------------------------------------------------
#Main Function

    def main(self, do_full_plot = True, save_data = True):
        print("Start analyzing: " + self.isotope + " " + self.desc + " ")
        print("Messured using: " + self.tool)
        
        #test if peaks have been found and either start peak_finder or continue
        if self.peaks_found == False:
            self.peak_finder()
            return
        else:
            self.load_peak_data()
            
        # restriced peak selection
        if self.peak_restriction:
            self.peak_pos = [ self.peak_pos[i] for i in self.restriced_peaks]
            self.peak_widths = [ self.peak_widths[i] for i in self.restriced_peaks ]
        
        

        # Loop over each peak
        for i in range(len(self.peak_pos)):
            center = self.peak_pos[i]
            width = int(self.peak_widths[i])  
            
            # Calculate the left and right borders and convert to int
            left_border = center - width // 2
            right_border = center + width // 2
            
            print("\n ###",i , " Pos: " , center)
            
            self.peaks.append(Peak(self.isotope, self.desc + str(i) , 100,
                              self.spectrum, self.peak_pos[i], left_border, right_border))
            
            self.peaks[i].do_fit(do_plot = True)
            

        if do_full_plot:
            self.full_plot(save = True)
        
        if save_data:
            self.save_spectrum_results()
                









































