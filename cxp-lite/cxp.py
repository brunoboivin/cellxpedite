import numpy as np
import pandas as pd
import os
import sys
import csv
from collections import OrderedDict
import Tkinter as tk
from Tkinter import *
import tkFileDialog, tkFont
import tkMessageBox
import tifffile as tiff
import scipy.misc
import shutil
import cv2
import random


def extract_timeseries(input_file, mask_file, output_dir, output_basename):
	# read image stack
	img_stack = tiff.imread(input_file)
	# flip axes order to have (x,y,z) instead of (z,x,y) where z: layer in stack
	img_stack = np.moveaxis(img_stack, 0, -1)

	# read mask
	mask = scipy.misc.imread(mask_file)

	# map (ob_num-->points)
	d = {}
	for i in np.unique(mask[mask > 0]):
		d[i] = np.nonzero(mask == i)

	# initialize pixel time series
	pixelsTimeSeries = []
	for k in sorted(d.keys()):
		objectKpixels = np.zeros((len(d[k][0]),img_stack.shape[-1]))
		pixelsTimeSeries.append(objectKpixels)
	pixelsTimeSeries = np.asarray(pixelsTimeSeries)

	# get objects batch time series
	ts_batch = [img_stack[d[k]] for k in sorted(d.keys())]
	for o,obj in enumerate(ts_batch):
		for p,pixel_batch_ts in enumerate(obj):
			pixelsTimeSeries[o][p] = pixel_batch_ts

	# individual object (aka cell) time series
	fragmentsTimeSeries = [
		np.mean(
			[ts for ts in pixelsTimeSeries[i]],
			axis=0
			)
			for i in range(len(d.keys()))
	]

	# Extract background fluorescence
	# randomly select background pixels
	numPixels = 10000  # number of random pixels
	background_coords = np.nonzero(mask == 0)

	# ensure sample size is <= population size
	if len(background_coords[0]) < numPixels:
	    numPixels = len(background_coords[0])

	# randomly select numPixels of indices
	randIndices = random.sample(range(len(background_coords[0])), numPixels)
	background_coords = (background_coords[0][randIndices],background_coords[1][randIndices])

	# initialize background time series
	backgroundTimeSeries = np.zeros((numPixels,len(fragmentsTimeSeries[0])))

	# compute background signal
	ts_background_batch = img_stack[background_coords]
	for i, bg_pixel_batch_ts in enumerate(ts_background_batch):
		backgroundTimeSeries[i] = bg_pixel_batch_ts

	background_intensity = np.mean([ts for ts in backgroundTimeSeries], axis=0)
	return fragmentsTimeSeries, background_intensity



def plotTimeSeries(ts,output_dir,output_basename, outputfile_suffix='timeseries', plot_title="object time series", xlabel='Frame #', ylabel='Fluorescence Intensity (a.u.)', xlim=None, ylim=None):
	import matplotlib.pyplot as plt
	
	def generateFigure(title, output_path, timeseries, bg=None, xlabel=xlabel, ylabel=ylabel):
		fig = plt.figure(figsize=(11, 8))
		for ts in timeseries:
			plt.plot(ts)
		if bg != None:
			plt.plot(bg, c='black', linewidth=3.0)
		plt.title(title, fontsize=16)
		plt.xlabel(xlabel, fontsize=14)
		plt.ylabel(ylabel, fontsize=14)
		plt.legend(prop={'size': 14})
		if xlim != None:
			plt.xlim(xlim)
		if ylim != None:
			plt.ylim(ylim)
		fig.savefig(output_path)
		plt.close(fig)

	output_filename = output_basename + "_" + outputfile_suffix + ".png"
	generateFigure(
		title=plot_title,
		output_path=os.path.join(output_dir, output_filename),
		timeseries=ts
	)


def find_peaks(x, min_peak_height=None, edge='rising', min_rel_height_neighbor=0, min_peak_distance=1, keep_peaks_same_height=False):
	# need at least 3 points to identify valid peaks
	if x.size < 3:
		return np.array([], dtype=int)

	# identify the different types of peaks
	dx = np.diff(x)
	singlePointPeaks, risingEdgePeaks, fallingEdgePeaks = np.array([[], [], []], dtype=int)
	if not edge:
		singlePointPeaks = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
	else:
		if edge.lower() in ['rising', 'both']:
			risingEdgePeaks = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
		if edge.lower() in ['falling', 'both']:
			fallingEdgePeaks = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
	ind = np.unique(np.hstack((singlePointPeaks, risingEdgePeaks, fallingEdgePeaks)))

	# first and last values of x cannot be peaks
	if ind.size and ind[0] == 0:
		ind = ind[1:]
	if ind.size and ind[-1] == x.size - 1:
		ind = ind[:-1]

	# keep only peaks > minimum peak height
	if ind.size and min_peak_height is not None:
		ind = ind[x[ind] >= min_peak_height]

	# remove peaks that are less than "neighbor_threshold" higher than their neighbors
	if ind.size and min_rel_height_neighbor > 0:
		dx_neighbors = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
		ind = np.delete(ind, np.where(dx_neighbors < min_rel_height_neighbor)[0])

	# identify peaks closer to one another than min_peak_distance
	if ind.size and min_peak_distance > 1:
		ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
		idel = np.zeros(ind.size, dtype=bool)
		for i in range(ind.size):
			if not idel[i]:
				# keep peaks with the same height if kpsh is True
				idel = idel | (ind >= ind[i] - min_peak_distance) & (ind <= ind[i] + min_peak_distance) & (x[ind[i]] > x[ind] if keep_peaks_same_height else True)
				idel[i] = 0  # Keep current peak
		# remove the small peaks and sort back the indexes by their occurrence
		ind = np.sort(ind[~idel])

	return ind


def smooth_signal(s,span=3):
    num_points = len(s)
    n = int((span-1)/2)
    if num_points < span:
        return s
    for i,x in enumerate(s[:-n]):
        if i < n:
            continue
        window = s[i-n:i+1+n]
        s[i] = np.mean(window)
    return s


def normalize_signal(s):
    s = np.array(s)
    return (s - min(s))/(max(s)-min(s))


def compute_signal_baseline(s, percentile_cutoff=10):
	baseline = np.percentile(s, percentile_cutoff)
	return baseline


def plotIndividualSignal(s, s_id, output_dir, filename_prefix='cell_', plt=None, peaks=None):
    plt.figure()
    plt.plot(s)
    plt.title("Cell {0}".format(s_id))
    if peaks != None:
    	plt.scatter(peaks, s[peaks], c='red')
    plt.savefig(os.path.join(output_dir,filename_prefix + "{0}.png".format(s_id)))
    plt.close()


def plotIndividualDetrendedSignal(s, s_detrended, s_id, output_dir, filename_prefix='cell_', 
	plt=None, peaks=None, 
	x_lim_raw=None, y_lim_raw=None, x_lim_dff=None, y_lim_dff=None):
	fig, axes = plt.subplots(2, sharex=True)
	fig.suptitle("Cell {0}".format(s_id))
	s_detrended = np.clip(s_detrended,0,max(s_detrended))

	if peaks != None:
		axes[1].scatter(peaks, s_detrended[peaks], c='red')

	# raw
	axes[0].plot(s)
	axes[0].set_ylabel('Fluorescence Intensity (a.u.)')
	if x_lim_raw != None:
		axes[1].set_xlim(x_lim_raw)
	if y_lim_raw != None:
		axes[1].set_ylim(y_lim_raw)

	# dff
	axes[1].plot(s_detrended)
	if x_lim_dff != None:
		axes[1].set_xlim(x_lim_dff)
	if y_lim_dff != None:
		axes[1].set_ylim(y_lim_dff)
	axes[1].set_xlabel('Frame #')
	axes[1].set_ylabel('DF/F')
	
	fig.savefig(os.path.join(output_dir,filename_prefix + "{0}.png".format(s_id)))
	plt.close()


def get_peak_indices(s, window_size=151, min_peak_height=2, min_peak_distance=50, noise_threshold=0.08):
    # smooth and clip differenced signal to 0 [diff => x(t) = x(t) - x(t-1)]
    # (this removes one from all indices as first data point is dropped, -1)
    signal = smooth_signal(np.diff(s), span=window_size)
    signal = np.clip(signal, 0, max(signal))
    
    # define window half-size as n, where window_size = 2*n + 1
    n = int((window_size-1)/2)
    
    # for peak identification purposes, we do not consider the first and last n datapoints
    trimmed_signal = np.array(signal[n:-n])    
    
    # identify peaks in trimmed_signal
    peak_indices = find_peaks(trimmed_signal, min_peak_height=min_peak_height, min_peak_distance=min_peak_distance)
    
    # adjust peak to local max using original signal
    # Step 1: compensate indices for trimming and differencing
    peak_indices = peak_indices + window_size  # (window_size-1) + 1
    # Step 2: update peak location to local max in peak region (i.e. peak +- n)
    peak_indices = [
        pid + np.argmax(s[pid-n:pid+n]) - n
        for pid in peak_indices
    ]
    
    # remove duplicates and sort peaks
    peak_indices = np.array(sorted(np.unique(peak_indices)))
    
    # peak filtering
    if len(peak_indices) > 1:
        filtered_peak_indices = set()
        for i,diff in enumerate(np.abs(np.diff(peak_indices))):
            first, second = peak_indices[i:i+2]

            if diff > min_peak_distance:
                filtered_peak_indices.add(first)

                # add last element of array of peak indices
                if i == len(np.diff(peak_indices))-1:
                    filtered_peak_indices.add(second)
            else:
                if np.argmax([first, second]) == 0:
                    filtered_peak_indices.add(first)
                else:
                    filtered_peak_indices.add(second)
    else:
        filtered_peak_indices = set(peak_indices)

    # reorder peak indices and filter out those below noise threshold or within last 100 data points
    filtered_peak_indices = [p for p in sorted(list(filtered_peak_indices)) 
                             if (s[p] > noise_threshold) and (p < len(s) - 100)]
    
    return filtered_peak_indices


def get_baseline(s, window=500, quantile=0.05, decreasing_only=True):
	# convert to df format so we can use the rolling function below
	roll_perc = pd.Series(s).rolling(window).quantile(quantile, interpolation='midpoint')

	# ensure rolling baseline is monotonically decreasing
	if decreasing_only:
		for i in range(len(roll_perc)-1):
			if roll_perc[i+1] > roll_perc[i]:
				roll_perc[i+1] = roll_perc[i]

	# ensure baseline doesn't equal 0
	baseline = roll_perc.values
	baseline = np.clip(baseline, 0, np.max(baseline)) + 1
	return baseline


def getPlotLimits():
	def tuplelize(s):
		try:
			axis_min, axis_max = s.split(',')
			return (float(axis_min), float(axis_max))
		except:
			return None

	x_lim_raw = None if xaxis_raw_entry.get().strip() == '' else tuplelize(xaxis_raw_entry.get())
	y_lim_raw = None if yaxis_raw_entry.get().strip() == '' else tuplelize(yaxis_raw_entry.get())
	x_lim_dff = None if xaxis_dff_entry.get().strip() == '' else tuplelize(xaxis_dff_entry.get())
	y_lim_dff = None if yaxis_dff_entry.get().strip() == '' else tuplelize(yaxis_dff_entry.get())
	return x_lim_raw, y_lim_raw, x_lim_dff, y_lim_dff


def extractFeatures(ts, ts_bg, output_dir, output_basename):
	# use initial value of save_signal_checkbox
	savesignals = bool(savesignal_checkbox_state.get())
	
	if savesignals:
		import matplotlib.pyplot as plt

		# output dir for cell signals
		output_dir_signals = os.path.join(output_dir,'individual_traces')
		if not os.path.exists(output_dir_signals):
			os.makedirs(output_dir_signals)		

	# Step 0: Retrieve user specified inputs
	smoothing_span = int(smoothing_span_entry.get())
	baseline_window = int(baseline_window_entry.get())
	baseline_quantile = float(baseline_quantile_entry.get())
	baseline_smooth_span = int(baseline_smoothing_entry.get())
	firstk = int(firstk_entry.get())

	# initialize output variables
	peaks = np.zeros(len(ts))
	amplitudes = np.zeros(len(ts))
	aucs = np.zeros(len(ts))
	dff_signals = []
	# retrieve user-specified x & y plot limits
	x_lim_raw, y_lim_raw, x_lim_dff, y_lim_dff = getPlotLimits()

	for i,s in enumerate(ts):
		s_id = i+1
		s = smooth_signal(s,smoothing_span)

		# compute baseline as running percentile over a fixed-size window
		baseline = get_baseline(s, window=baseline_window, quantile=baseline_quantile, decreasing_only=False)

		# trim off first k data points
		s = s[firstk:]
		baseline = baseline[firstk:]
		baseline = smooth_signal(baseline, span=baseline_smooth_span)

		# subtract baseline and normalize by it
		signal_detrended = (s - baseline) / baseline
		dff_signals.append(signal_detrended)

		# identify peaks
		new_peak_indices = get_peak_indices(
			signal_detrended,
			window_size=int(peak_window_entry.get()), 
			min_peak_height=float(min_peak_height_entry.get()), 
			min_peak_distance=int(min_peak_dist_entry.get()),
			noise_threshold=float(noise_threshold_entry.get())
		)

		# plot signal
		if savesignals:
			plotIndividualDetrendedSignal(s, signal_detrended, s_id, output_dir_signals, 
				plt=plt, peaks=new_peak_indices,
				x_lim_raw=x_lim_raw, y_lim_raw=y_lim_raw,
				x_lim_dff=x_lim_dff, y_lim_dff=y_lim_dff)

		peaks[i] = len(new_peak_indices)
		amplitudes[i] = np.mean(signal_detrended[new_peak_indices]) if len(new_peak_indices) > 0 else 0
		aucs[i] = np.trapz( np.clip(signal_detrended, 0, max(signal_detrended)) )

	# save original signals (new format - Oct 2019)
	df_ts_file = output_basename + '_timeseries_original.csv'
	df_ts = pd.DataFrame()
	df_ts['background'] = ts_bg
	for i,s in enumerate(ts):
		df_ts['cell_'+str(i+1)] = s
	df_ts.to_csv(os.path.join(output_dir, df_ts_file), index=False)

	# save df/f signals
	df_ts_file = output_basename + '_timeseries_dff.csv'
	df_ts = pd.DataFrame()
	for i,s in enumerate(dff_signals):
		df_ts['cell_'+str(i+1)] = s
	df_ts.to_csv(os.path.join(output_dir, df_ts_file), index=False)

	# save plot of selected signals and dff signals
	plotTimeSeries(
		[s for i,s in enumerate(ts)], 
		output_dir, 
		output_basename, 
		outputfile_suffix='timeseries_original', 
		plot_title="Original Fluorescence Traces", 
		ylabel="Fluorescence Intensity (a.u.)",
		xlim=x_lim_raw,
		ylim=y_lim_raw
		)
	plotTimeSeries(
		dff_signals, 
		output_dir, 
		output_basename, 
		outputfile_suffix='timeseries_dff', 
		plot_title="DF/F Fluorescence Traces", 
		ylabel="DF/F (a.u.)",
		xlim=x_lim_dff,
		ylim=y_lim_dff
		)

	# construct grid (df) containing features
	data_dict = OrderedDict()
	data_dict['well'] = output_basename
	data_dict['cell id'] = list(range(1,len(ts)+1))
	data_dict['peak count'] = peaks.tolist()
	data_dict['amplitude'] = amplitudes.tolist()
	data_dict['auc'] = aucs.tolist()
	df = pd.DataFrame(data_dict)

	# save to csv file
	return df


####### GUI ###########
# Global vars
global input_folder
input_folder = ''

# helper functions
def selectInputFolder():
	selectedFolder = tkFileDialog.askdirectory(title="Select input folder", initialdir='.')
	if selectedFolder:
		global input_folder
		input_folder = selectedFolder
		input_folder_basename = os.path.basename(selectedFolder)
		inputSelectedVar.set(input_folder_basename[:65])

def start():
	try:
		global input_folder

		# disable start button to prevent multiple clicks
		startButton.config(state="disabled")

		# ensure input folder is specified
		if input_folder == '' or input_folder == 'no folder selected': 
			tkMessageBox.showwarning("Missing user input", "An input folder must be selected.")
			return

		# identify image files
		mask_suffix = '_mask'
		mask_file_ext = '.tiff'
		img_files = [os.path.join(input_folder,f) for f in os.listdir(input_folder)
				if (os.path.isfile(os.path.join(input_folder,f)) 
					and f != '.DS_Store'
					and mask_suffix not in f)]

		# ensure input folder is not empty
		if len(img_files) == 0: 
			tkMessageBox.showwarning("Missing data", "The input folder is empty.")
			return

		# match (img_stack, mask); ensure all mask files exist
		data = []
		for img_file in img_files:
			img_filename, img_file_extension = os.path.splitext(img_file)
			mask_file = img_filename + mask_suffix + mask_file_ext
			if os.path.isfile(mask_file):
				data.append((img_file, mask_file))
			else:
				tkMessageBox.showwarning("Missing data", "Missing mask file: \n\n" + mask_file)
				return

		# analyze each pair of files (img stack, mask)
		errors = False
		for img_file, mask_file in data:
			try:
				# create new output directory
				output_basename = os.path.basename(os.path.splitext(img_file)[0])
				output_dir = os.path.join(input_folder, 'output_'+output_basename)
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)

				# extract time series
				fragmentsTimeSeries, background_intensity = extract_timeseries(img_file, mask_file, output_dir, output_basename)

				# compute time series features
				df_features = extractFeatures(fragmentsTimeSeries, background_intensity, output_dir, output_basename)

				# save df_features
				df_features.to_csv(os.path.join(output_dir, output_basename + "_features.csv"), index=False)
			except:
				errors = True

		# output message
		if errors:
			output_status = 'Complete'
			output_message = 'Data extraction completed, but some files could not be analyzed.'
			tkMessageBox.showwarning(output_status, output_message)
		else:
			output_status = 'Success'
			output_message = 'Data extraction completed successfully.'
			tkMessageBox.showinfo(output_status, output_message)

	except Exception as e:
		# display error to user and console
		import traceback
		print(traceback.format_exc())
		tkMessageBox.showwarning("Error", "An error occured during the analysis:\n\n" + str(e))
	finally:
		# re-enable start button
		startButton.config(state="normal")


# GUI frames
# params for interface
bgColor = '#d8d8d8'
btnWidth = 12
xpadding = (10,5)

# initialize interface
root = tk.Tk()
root.title('CXP')
root.resizable(width=False, height=False)
root.geometry('545x290')  # width x height
root.config(bg=bgColor)

# fonts
font_buttons = tkFont.Font(root=root, family='Arial', size=14, weight='bold')
font_labels = tkFont.Font(root=root, family='Arial', size=14, weight='bold')
font_output = tkFont.Font(root=root, family='Arial', size=14)
font_filenames = tkFont.Font(root=root, family='Arial', size=12)

# MAIN FRAME
mainFrame = Frame(root, bg=bgColor, padx=10, pady=15)

# input file selection
selectFileBtn = Button(mainFrame, text='Select folder', command=selectInputFolder, highlightbackground=bgColor, font=font_buttons, width=btnWidth)
selectFileBtn.grid(row=0, column=0, sticky=W)
# label for selected folder
inputSelectedVar = StringVar()
inputSelectedVar.set('no folder selected') 
selectedFileLabel = Label(mainFrame, textvariable=inputSelectedVar, bg=bgColor, font=font_filenames)
selectedFileLabel.grid(row=0, column=1, columnspan=5, sticky=W, padx=xpadding)

# analysis params
firstk_label = Label(mainFrame, text="Start at frame", bg=bgColor, font=font_output)
firstk_label.grid(row=3, column=0, sticky=W)
firstk_entry = Entry(mainFrame, width=btnWidth)
firstk_entry.insert(END, '500')
firstk_entry.grid(row=3, column=1, sticky=W, padx=xpadding)

# smoothing params
smoothing_span_label = Label(mainFrame, text="Signal smoothing", bg=bgColor, font=font_output)
smoothing_span_label.grid(row=4, column=0, sticky=W)
smoothing_span_entry = Entry(mainFrame, width=btnWidth)
smoothing_span_entry.insert(END, '11')
smoothing_span_entry.grid(row=4, column=1, sticky=W, padx=xpadding)

# axes' parameters
xaxis_raw_label = Label(mainFrame, text="X-axis limits (Raw)", bg=bgColor, font=font_output)
xaxis_raw_label.grid(row=5, column=0, sticky=W)
xaxis_raw_entry = Entry(mainFrame, width=btnWidth)
xaxis_raw_entry.insert(END, '')
xaxis_raw_entry.grid(row=5, column=1, sticky=W, padx=xpadding)

yaxis_raw_label = Label(mainFrame, text="Y-axis limits (Raw)", bg=bgColor, font=font_output)
yaxis_raw_label.grid(row=6, column=0, sticky=W)
yaxis_raw_entry = Entry(mainFrame, width=btnWidth)
yaxis_raw_entry.insert(END, '')
yaxis_raw_entry.grid(row=6, column=1, sticky=W, padx=xpadding)

xaxis_dff_label = Label(mainFrame, text="X-axis limits (DFF)", bg=bgColor, font=font_output)
xaxis_dff_label.grid(row=7, column=0, sticky=W)
xaxis_dff_entry = Entry(mainFrame, width=btnWidth)
xaxis_dff_entry.insert(END, '')
xaxis_dff_entry.grid(row=7, column=1, sticky=W, padx=xpadding)

yaxis_dff_label = Label(mainFrame, text="Y-axis limits (DFF)", bg=bgColor, font=font_output)
yaxis_dff_label.grid(row=8, column=0, sticky=W)
yaxis_dff_entry = Entry(mainFrame, width=btnWidth)
yaxis_dff_entry.insert(END, '')
yaxis_dff_entry.grid(row=8, column=1, sticky=W, padx=xpadding)

# baseline params
baseline_fg_color = '#de5c00'
baseline_window_label = Label(mainFrame, text="Baseline window", bg=bgColor, font=font_output, fg=baseline_fg_color)
baseline_window_label.grid(row=3, column=2, sticky=W)
baseline_window_entry = Entry(mainFrame, width=btnWidth)
baseline_window_entry.insert(END, '100')
baseline_window_entry.grid(row=3, column=3, sticky=W, padx=xpadding)

baseline_quantile_label = Label(mainFrame, text="Baseline quantile", bg=bgColor, font=font_output, fg=baseline_fg_color)
baseline_quantile_label.grid(row=4, column=2, sticky=W)
baseline_quantile_entry = Entry(mainFrame, width=btnWidth)
baseline_quantile_entry.insert(END, '0.05')
baseline_quantile_entry.grid(row=4, column=3, sticky=W, padx=xpadding)

baseline_smoothing_label = Label(mainFrame, text="Baseline smoothing", bg=bgColor, font=font_output, fg=baseline_fg_color)
baseline_smoothing_label.grid(row=5, column=2, sticky=W)
baseline_smoothing_entry = Entry(mainFrame, width=btnWidth)
baseline_smoothing_entry.insert(END, '100')
baseline_smoothing_entry.grid(row=5, column=3, sticky=W, padx=xpadding)

# analysis params
peak_fg_color = 'blue'
noise_threshold_label = Label(mainFrame, text="Peak threshold", bg=bgColor, font=font_output, fg=peak_fg_color)
noise_threshold_label.grid(row=6, column=2, sticky=W)
noise_threshold_entry = Entry(mainFrame, width=btnWidth)
noise_threshold_entry.insert(END, '0.05')
noise_threshold_entry.grid(row=6, column=3, sticky=W, padx=xpadding)

min_peak_dist_label = Label(mainFrame, text="Peak distance", bg=bgColor, font=font_output, fg=peak_fg_color)
min_peak_dist_label.grid(row=7, column=2, sticky=W)
min_peak_dist_entry = Entry(mainFrame, width=btnWidth)
min_peak_dist_entry.insert(END, '50')
min_peak_dist_entry.grid(row=7, column=3, sticky=W, padx=xpadding)

peak_window_label = Label(mainFrame, text="Peak window", bg=bgColor, font=font_output, fg=peak_fg_color)
peak_window_label.grid(row=8, column=2, sticky=W)
peak_window_entry = Entry(mainFrame, width=btnWidth)
peak_window_entry.insert(END, '100')
peak_window_entry.grid(row=8, column=3, sticky=W, padx=xpadding)

min_peak_height_label = Label(mainFrame, text="Peak height", bg=bgColor, font=font_output, fg=peak_fg_color)
min_peak_height_label.grid(row=9, column=2, sticky=W)
min_peak_height_entry = Entry(mainFrame, width=btnWidth)
min_peak_height_entry.insert(END, '0.00025')
min_peak_height_entry.grid(row=9, column=3, sticky=W, padx=xpadding)

# checkbox for saving signals from active cells
savesignal_checkbox_state = IntVar()
savesignal_checkbox = Checkbutton(mainFrame, text="Save individual traces", variable=savesignal_checkbox_state, bg=bgColor, font=font_buttons)
savesignal_checkbox.grid(row=9, column=0, columnspan=2, sticky=W)  # pady=(5,0)

# start analysis
startButton = Button(mainFrame, text='Start', command=start, highlightbackground=bgColor, font=font_buttons, width=btnWidth)
startButton.grid(row=13, column=0, pady=10, sticky=W)
# exit button
exitButton = Button(mainFrame, text='Exit', command=root.quit, highlightbackground=bgColor, font=font_buttons, width=btnWidth)
exitButton.grid(row=13, column=1, pady=10, sticky=E)

# pack main frame
mainFrame.pack(fill='both')

# tmp fix for buttons not displaying properly
def fix():
    a = root.winfo_geometry().split('+')[0]
    b = a.split('x')
    w = int(b[0])
    h = int(b[1])
    root.geometry('%dx%d' % (w+1,h+1))
root.update()
root.after(0, fix)

# start gui
root.mainloop()
