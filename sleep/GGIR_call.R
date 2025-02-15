library(GGIR)

data_dir <- "/Users/marcellosicbaldi/PPG_pipelines/GGIR_input/"
output_dir <- "/Users/marcellosicbaldi/PPG_pipelines/GGIR_output/"
sleeplog_dir <- "/Users/marcellosicbaldi/PPG_pipelines/GGIR_input/sleeplog/"
GGIR(
  datadir = data_dir,
  outputdir = output_dir,
  #loglocation = sleeplog_dir,
  sleepwindowType = "SPT", 
  HASIB.algo = "vanHees2015", # default SIB detection
  #=====================
  # read.myacc.csv arguments
  #=====================
  rmc.firstrow.acc = 2, # first ACC column
  rmc.col.time = 1, # time columns
  rmc.col.acc = 2:4, # ACC columns
  rmc.unit.acc = "mg", # ACC units
  rmc.format.time = "%Y-%m-%d %H:%M:%S", # format of Empatica timestamp
  rmc.unit.time = "POSIX",
  rmc.sf = 64, # Empatica sampling frequency
  # rmc.origin = "1970-01-01", # origin of Empatica timestamp
)