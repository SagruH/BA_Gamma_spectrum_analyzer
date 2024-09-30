"""
All Depencies:
    Numpy
    Pandas
    scipy
    matplotlib
    yaml
    PhyPraKit


"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

import PhyPraKit as ppk

from peak_class import Peak
from spectrum_class import Spectrum
from scipy.interpolate import interp1d

#-----------------------------------------------------------------
#book values

#values
cs137_energy = [662]        #kev #channel num 265
co60_energy  = [1173, 1333] #kev #            452,509
na22_energy  = [511, 1275]  #kev #            209, 489


def poly2nd(x,a = 1, b = 1, c = 0):
    return a*x**2 + b*x + c 


#-----------------------------------------------------------------
#functions

def load_json_files_from_folder(folder_path):
    """
    Loads all JSON files from the specified folder.
    
    :param folder_path: Path to the folder containing the JSON files.
    :return: List of dictionaries loaded from the JSON files.
    """
    json_data = []
    
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):  # Only consider .json files
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)  # Load the JSON content
                json_data.append(data)       # Store the loaded data in the list
                
    return json_data



def extract_values_from_spectrum_data(json_data):
    """
    Extracts FWHM, FWHM errors, and book values from loaded JSON data.
    
    :param json_data: List of JSON data (dictionaries) loaded from files.
    :return: Three lists - one for FWHM values, one for FWHM errors, and one for book values.
    """
    fwhm_values = []
    fwhm_errors = []
    book_values = []
    mu_values = []
    mu_errors = []
    
    for data in json_data:
        if 'peaks' in data:
            for peak in data['peaks']:
                for param in peak['parameters']:
                    # Check for FWHM and book value
                    if 'fwhm' in param and 'fwhm_error' in param:
                        fwhm_value = param['fwhm']
                        fwhm_error = param['fwhm_error']
                        fwhm_values.append(fwhm_value)
                        fwhm_errors.append(fwhm_error)
                        
                    if 'book_value' in param:
                        mu_values.append(param['value'])
                        mu_errors.append(param['error'])
                        book_value = param['book_value']
                        book_values.append(book_value)
    
    return [mu_values, mu_errors], [fwhm_values, fwhm_errors], book_values


def convert_to_latex_table(mu_data, fwhm_data, book_values):
    """
    Converts the extracted FWHM, FWHM errors, and book values into a LaTeX table format.
    Adds a horizontal line and a bold 'Isotop-Name' in the first column when the book value decreases.
    
    :param mu_data: List containing mu values and their errors.
    :param fwhm_data: List containing FWHM values and their errors.
    :param book_values: List of peak energy values (formerly book values).
    :return: LaTeX formatted table as a string.
    """
    mu_values, mu_errors = mu_data
    fwhm_values, fwhm_errors = fwhm_data

    # Start the LaTeX table without vertical lines
    latex_table = "\\begin{table}[ht]\n\\centering\n"
    latex_table += "\\caption{Peak-Energie, $\\mu$, and FWHM values.}\n"
    latex_table += "\\begin{tabular}{c c c c c}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\textbf{Isotop} & \\textbf{Peak-Energie (keV)} & \\textbf{Position} & \\textbf{FWHM} & \\textbf{FWHM (\%)} \\\\\n"
    
    # Initialize the previous book value
    previous_book_value = -1

    # Iterate over the data and fill the table
    for i in range(len(fwhm_values)):
        book_str = f"{book_values[i]:.2f}"
        mu_str = f"{mu_values[i]:.2f} $\\pm$ {mu_errors[i]:.2f}"
        fwhm_str = f"{fwhm_values[i]:.2f} $\\pm$ {fwhm_errors[i]:.2f}"

        # Calculate FWHM in percentage and its error
        fwhm_percent = (fwhm_values[i] / mu_values[i]) * 100
        fwhm_percent_str = f"{fwhm_percent:.2f}"
        
        # Calculate the error for FWHM in percentage
        # Error propagation formula for FWHM in percentage
        error_fwhm_percent = fwhm_percent * np.sqrt(
            (fwhm_errors[i] / fwhm_values[i])**2 + mu_errors[i]
        )
        error_fwhm_percent_str = f"{error_fwhm_percent:.2f}"

        # Add the new row to the table
        if previous_book_value is not None and book_values[i] < previous_book_value or previous_book_value == -1:
            # Add a horizontal line before the current row and insert 'Isotop-Name' in the first column
            latex_table += "\\midrule\n"
            latex_table += f"\\textbf{{Isotop-Name}} & {book_str} & {mu_str} & {fwhm_str} & {fwhm_percent_str} $\\pm$ {error_fwhm_percent_str} \\\\\n"
        else:
            # Add the current row without the Isotop-Name
            latex_table += f" & {book_str} & {mu_str} & {fwhm_str} & {fwhm_percent_str} $\\pm$ {error_fwhm_percent_str} \\\\\n"
        
        # Update the previous book value
        previous_book_value = book_values[i]
    
    # End the LaTeX table
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"
    
    return latex_table




#-----------------------------------------------------------------
#import data

# Radiacode    
# Frist measurements
csr1 = Spectrum("Cs137", "firstR", [662], "Radiacode", 'data_radiacode/m1Cs137_240226-1235.yaml', 
                restricted_peaks = [False,[0]])
cor1 = Spectrum("Co60", "firstR", [1], "Radiacode", 'data_radiacode/m2C060_240226-1246.yaml', 
                restricted_peaks = [False,[0]])
nar1 = Spectrum("Na22", "firstR", [1], "Radiacode", 'data_radiacode/m3Na22_240226-1256.yaml', 
                restricted_peaks = [False,[0]])
rar1 = Spectrum("Ra226", "firstR", [1], "Radiacode", 'data_radiacode/m4Ra226_240226-1303.yaml', 
                restricted_peaks = [False,[0]])

# Second measurements
csr2 = Spectrum("Cs137", "secondR", [662], "Radiacode", 'data_radiacode/r_CS137_20min.yaml',
                restricted_peaks = [True,[1]])
cor2 = Spectrum("Co60", "secondR", [1173.2, 1333.5], "Radiacode", 'data_radiacode/r_co60_20min.yaml', 
                restricted_peaks = [True,[2,3]])
nar2 = Spectrum("Na22", "secondR", [511, 1274.5], "Radiacode", 'data_radiacode/r_na22_20min.yaml', 
                restricted_peaks = [False,[0]])
rar2 = Spectrum("Ra226", "secondR", [186.1, 241.98, 295.21, 351.92, 609.31], "Radiacode", 'data_radiacode/r_ra226_20min.yaml', 
                restricted_peaks = [True,[1,2,3,4,5]])


# Lab
# Frist measurements
csl1 = Spectrum("Cs137", "firstL", [662], "Lab_old", 'data_lab/M2CS137.csv', 
                restricted_peaks = [True,[1]])
col1 = Spectrum("Co60", "firstL", [1173.2, 1333.5], "Lab_old", 'data_lab/M1Co60.csv', 
                restricted_peaks = [True,[3,4]])
nal1 = Spectrum("Na22", "firstL", [511, 1274.5], "Lab_old", 'data_lab/M3Ra226.csv', 
                restricted_peaks = [False,[0]])
ral1 = Spectrum("Ra226", "firstL", [1], "Lab_old", 'data_lab/M4Na22.csv', 
                restricted_peaks = [False,[1]])

# Second measurements
csl2 = Spectrum("Cs137", "secondL", [662], "Lab_new", 'data_lab/L_Cs137_1200s.hst',
                restricted_peaks = [True,[1]])
col2 = Spectrum("Co60", "secondL", [1173.2, 1333.5], "Lab_new", 'data_lab/L_Co60_1200s.hst', 
                restricted_peaks = [True,[2,3]])
nal2 = Spectrum("Na22", "secondL", [511, 1274.5], "Lab_new", 'data_lab/L_Na22_1200s.hst', 
                restricted_peaks = [True,[0,1]])
ral2 = Spectrum("Ra226", "secondL", [186.1, 241.98, 295.21, 351.92, 609.31], "Lab_new", 'data_lab/L_Ra223_1200s.hst', 
                restricted_peaks = [True,[2,3,4,5,6]])

#-----------------------------------------------------------------
#do code

#do main fuctions of spectrum classes
# csr2.main(do_full_plot = True, save_data = False)
# cor2.main(do_full_plot = True, save_data = False)
# nar2.main(do_full_plot = True, save_data = False)
# rar2.main(do_full_plot = True, save_data = False)


# csl2.main(do_full_plot = True, save_data = False)
# col2.main(do_full_plot = True, save_data = False)
# nal2.main(do_full_plot = True, save_data = False)
# ral2.main(do_full_plot = True, save_data = False)


#csl2.quick_plot(save = True)
#nal2.quick_plot(save = True)


#load jsons and extract data for radiacode run 2
radiacode_spectrum_data = load_json_files_from_folder("radiacode_spectrum_results")
rmu_val_err, rfwhm_val_err, rbook_values = extract_values_from_spectrum_data(radiacode_spectrum_data)


#load jsons and extract data for lab run 2
lab_spectrum_data = load_json_files_from_folder("lab_spectrum_results")
lmu_val_err, lfwhm_val_err, lbook_values = extract_values_from_spectrum_data(lab_spectrum_data)


#convert in latex table
#latex_table = convert_to_latex_table(rmu_val_err, rfwhm_val_err, rbook_values)
latex_table = convert_to_latex_table(lmu_val_err, lfwhm_val_err, lbook_values)

#do Channel Energy fit

# energyfitr = ppk.mnFit(fit_type="xy")
# energyfitr.init_xyData(rbook_values, rmu_val_err[0], ey=rmu_val_err[1])
# energyfitr.init_xyFit(poly2nd,p0=[1.,1.,1.])
# energyfitr.do_fit()
# # energyfitr.plotModel(axis_labels=['Energie in keV','Kanal'], model_legend= '$ax^2 + bx + c$')
# # plt.savefig('quick_pic/'+'linear_radiacode.png', dpi = 400)

# energyfitl = ppk.mnFit(fit_type="xy")
# energyfitl.init_xyData(lbook_values, lmu_val_err[0], ey=lmu_val_err[1])
# energyfitl.init_xyFit(poly2nd,p0=[1.,1.,1.])
# energyfitl.do_fit()
# # energyfitl.plotModel(axis_labels=['Energie in keV','Kanal'], model_legend= '$ax^2 + bx + c$')
# # plt.savefig('quick_pic/'+'linear_lab.png', dpi = 400)

energyfitr = ppk.mnFit(fit_type="xy")
energyfitr.init_xyData(rmu_val_err[0],rbook_values , ex=rmu_val_err[1])
energyfitr.init_xyFit(poly2nd,p0=[1.,1.,1.])
energyfitr.do_fit()
# energyfitr.plotModel(axis_labels=['Energie in keV','Kanal'], model_legend= '$ax^2 + bx + c$')
# plt.savefig('quick_pic/'+'linear_radiacode.png', dpi = 400)

energyfitl = ppk.mnFit(fit_type="xy")
energyfitl.init_xyData(lmu_val_err[0], lbook_values, ex=lmu_val_err[1])
energyfitl.init_xyFit(poly2nd,p0=[1.,1.,1.])
energyfitl.do_fit()
# energyfitl.plotModel(axis_labels=['Energie in keV','Kanal'], model_legend= '$ax^2 + bx + c$')
# plt.savefig('quick_pic/'+'linear_lab.png', dpi = 400)




#nachweiseffizienz finden
#channel to energy
paraml = energyfitl.getResult().get('parameter values')
paramr = energyfitr.getResult().get('parameter values')

energier = paramr[0] * rar2.channel**2 + paramr[1] * rar2.channel + paramr[2]
energiel = paraml[0] * ral2.channel**2 + paraml[1] * ral2.channel + paraml[2]
xenergy = np.linspace(min(min(energier), min(energiel)), max(max(energier), max(energiel)),  num=1000)

# Interpolate both spectra to the common energy grid
interp_counts_1 = interp1d(energier, rar2.spectrum, bounds_error=False, fill_value=0)(xenergy)
interp_counts_2 = interp1d(energiel, ral2.spectrum, bounds_error=False, fill_value=0)(xenergy)

# Calculate the divided spectrum
divided_spectrum = np.divide(interp_counts_1, interp_counts_2, out=np.zeros_like(interp_counts_1), where=interp_counts_2!=0)

# find xlim
xlim = max(xenergy)


#plt.plot(ral2.channel, rar2.spectrum/ral2.spectrum, color='black', label="$N_R$ / $N_L$", marker='x', linestyle='', markersize=1)
 
 # If peaks are to be plotted, highlight them with red markers

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(xenergy, interp_counts_1, label='Radiacode (Counts)', color='blue')
#plt.title('Interpolated Spectrum 1')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, xlim)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(xenergy, interp_counts_2, label='Current Setup (Counts)', color='orange')
#plt.title('Interpolated Spectrum 2')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, xlim)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(xenergy, divided_spectrum, label='Divided Spectrum (Counts)', color='green')
#plt.title('Divided Spectrum (Counts)')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts Radiacode / Counts Lab')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, xlim)
plt.legend()

plt.tight_layout()

plt.show()
 
 # Add labels, title and make pretty
# plt.xlim(0, xlim)
# plt.xlabel('Energy in keV', fontsize=12)
# plt.ylabel('$N_R$ / $N_L$', fontsize=12)
# #plt.title('Spectrum Plot' + " " + self.isotope + " " + self.desc)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()

plt.savefig('quick_pic/nachweis', dpi = 400)





#print(latex_table)

#Display the results
# print("\n\n\n")
# print("Mu Values and Errors:", mu_val_err)
# print("Book Values:", book_values)
# print("FWHM and Errors:", fwhm_val_err)



#plt.show()
print("End")













