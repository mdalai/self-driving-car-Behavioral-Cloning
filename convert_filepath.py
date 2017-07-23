import sys
import pandas as pd

#DATAPATH="data2/data/"
#DATAPATH="data0/"
#exit the code: sys.exit()

def convert_filepath(datapath):

	df = pd.read_csv(datapath + 'driving_log.csv', header=None)

	# Check if the filepath is already updated
	if df[0][1].find('\\') == -1:
		print(datapath + ': the filepath in "driving_log.csv" are ALREADY UPDATED!!!')
		return

	# Add Headers
	df.columns = ["center", "left", "right", "steering","throttle","brake","speed"]

	# Update file path
	df['center'] = df['center'].str.split("\\").str[-2:].str.join('/')
	df['left'] = df['left'].str.split("\\").str[-2:].str.join('/')
	df['right'] = df['right'].str.split("\\").str[-2:].str.join('/')

	# Save the file, replace the existing file
	df.to_csv(datapath + 'driving_log.csv', index=False)

	print(datapath + ': Successfully CONVERTED!!!')

if __name__ == '__main__':
    for pathname in sys.argv[1:]:
        convert_filepath(pathname)