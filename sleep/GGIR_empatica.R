library(GGIR)

data_dir <- "/Users/augenpro/Documents/Empatica/data_sara/data/GGIR_input_new"
output_dir <- "/Users/augenpro/Documents/Empatica/data_sara/data/GGIR_output_new"

GGIR(
  
  datadir = data_dir,  # paths to acc CSV data files
  outputdir= output_dir, # path where data are exported
  
  mode=c(1:5), # which of the five parts should be run?
  
  # !!! changed from c(2,4,5) to save computational time
  do.report=c(), # summary spreadsheet generation? 
  # !!!
  
  studyname = "DARE3.3_pilot", # name of the study
  overwrite = FALSE, # overwrite analysis for which milestone data already exists?
  do.imp=TRUE, # imputing missing values by g.impute in GGIR g.part2
  # !!! Changed from FALSE to save computational time
  do.parallel = TRUE, # Whether to use multi-core processing
  # !!!
  print.filename=TRUE, # Whether to print the filename before analysing it
  storefolderstructure = FALSE, # Whether storing folder structure of the acc data
  printsummary=TRUE, # Print summary of calibration procedure in the console when done?
  epochvalues2csv=TRUE, # reporting epoch values to a CSV file?
  minimum_MM_length.part5 = 12, # Min n.hours of MM day to be included in cleaned g.part5
  
  #.................................................
  # read.myacc.csv arguments
  #.................................................
  # !!! Changed from 4 because otherwise it only takes the first 4 rows ?
  rmc.nrow = Inf, # Number of rows to read from the ACC file. Here, 4 (time,x,y,z)
  # !!!
  rmc.dec = ".", # Decimal used for numbers
  rmc.firstrow.acc = 2, # First row (number) of the acceleration data (default = NULL)
  rmc.col.acc = 2:4, # Vector with 3 column (numbers) in which the acc signals are stored
  rmc.col.time = 1, # Scalar with column (number) in which the timestamps are stored
  rmc.unit.acc = "mg", # Character with unit of acceleration values ("g", "mg", or "bit")
  rmc.unit.time = "POSIX", # Character with unit of timestamps (default POSIX)
  rmc.format.time = "%Y-%m-%d %H:%M:%S", # date-time format as used by strptime
  # !!! Changed from "Europe/Brussels" to the default (i.e., system time zone)
  desiredtz = "", # Timezone in which experiments took place (default = "")
  # !!!
  # !!! Changed from NULL to "" (i.e., system time zone)
  configtz = "", # Timezone in which device was configured (dafault = "")
  # !!!
  rmc.sf = 64, # Sample rate in Hertz
  rmc.noise = 13, # Noise level of acc (mg) used with ad-hoc .csv data formats (???)
  # rmc.col.temp=5, # to add temperature
  

  #.................................................
  # Part 2
  #.................................................
  # strategy = 1, # How to deal with knowledge about study protocol (default: 1)
  # ndayswindow=3, # argument working only with strategy 3 o 5
  hrs.del.start = 0, # n. HOURS between start of experiment and wearing of monitor
  hrs.del.end = 0, # n. HOURS between unwearing the monitor and end of the experiment
  cosinor = TRUE, # Whether to apply the cosinor analysis from the ActCR package
  maxdur = 9, # n. DAYS between start and end of the experiment
  includedaycrit = 16, # min n. of valid HOURS in calendar day specific to analysis in part 2
  L5M5window = c(0,12), # start and end time on which L5M5 needs to be calculated
  M5L5res = 10, # Resolution of L5 and M5 analysis in minutes (default=10)
  winhr = c(5,10), # Vector of window size(s) (unit: hours) of LX and MX analysis (default 5)
  qlevels = c(c(1380/1440),c(1410/1440)), # percentiles for which value needs to be extracted
  qwindow=c(0,24), # windows over which all variables are calculated
  ilevels = c(seq(0,400,by=50),8000), # Levels for acc value (mg) frequency distribution
  mvpathreshold =c(100), # Acceleration threshold for MVPA estimation in GGIR g.part2
  
  # !!! changed from TRUE to save computational time
  do.part2.pdf=TRUE, # Whether to generate a pdf for g.part2
  # !!!
  
  #.................................................
  # Part 3 + 4
  #.................................................
  timethreshold = c(5), # threshold (min) for sustained inactivity periods (default 5)
  anglethreshold = 5, # threshold (°) for sustained inactivity periods (default 5)
  ignorenonwear = TRUE, # If TRUE then ignore detected monitor non-wear periods
  
  # !!! changed from TRUE to save computational time
  do.visual = TRUE, # pdf visuals of overlap between sleeplog and acc
  # !!!
  
  outliers.only = FALSE, # if FALSE include all available nights in the visual pdf report
  # criterror = 3, # only if do.visual=TRUE & outliers.only=TRUE
  
  # Part 4 parameters:
  #...........................
  excludefirstlast = FALSE, # ignore the first and last night
  
  # !!! changed from 16 because of shorter recordings in our protocol
  includenightcrit = 10, # min no. of valid hours per night (24h window between noon and noon)
  # !!!
  
  def.noc.sleep = 1, # time window during which sustained inactivity will be assumed to represent sleep (default=1) - only used if no sleep log entry is available
  
  # !!! no sleep log --> HDCZA
  loglocation=NULL, # Path to sleep log csv (default NULL)
  sleepwindowType="SPT", # type of info in the sleeplog, "SPT" for sleep period time, "TimeInBed" if sleep log recorded time in bed
  # !!!
  
  relyonguider = TRUE, # whether SIB that overlap with the guider are labelled as sleep
  
  #.................................................
  # Part 5
  #.................................................
  # !!! changed from c(30) to c(40), which is the default
  threshold.lig = c(40), # Threshold for light PA to separate from inactivity (def 40)
  # !!!
  threshold.mod = c(100), # Threshold for moderate PA to separate from light (def 100)
  threshold.vig = c(400), # Threshold for vigorous PA to separate from moderate (def 400)
  boutcriter = 0.8, # what fraction of a bout should be above the mvpathreshold (def .8)
  boutcriter.in = 0.9, # fraction of a bout should be below the threshold.lig (def .9)
  boutcriter.lig = 0.8, # fraction that should be between threshold.lig and threshold.mod (def 0.8)
  boutcriter.mvpa = 0.8, # fraction of a bout needs to be above the threshold.mod (def c(10,20,30))
  boutdur.in = c(10,20,30), # duration(s) (min) of inactivity bouts to be extracted (def c(10, 20, 30))
  boutdur.lig = c(1,5,10), # duration(s) (min) of light activity bouts to be extracted (def c(1,5,10))
  boutdur.mvpa = c(1,5,10), # duration(s) of MVPA bouts in minutes (def c(1, 5, 10))
  includedaycrit.part5 = 2/3, # Inclusion criteria for no. of valid waking hours (def 2/3)
  frag.metrics="all", # Fragmentation metric to extract
  save_ms5rawlevels = TRUE, # Save the time series classification (levels)
  save_ms5raw_format = "csv", # how data should be stored (csv or RData)
  part5_agg2_60seconds=TRUE, # use aggregate epochs to 60 sec as part of the g.part5 analysis
  save_ms5raw_without_invalid=FALSE, # remove invalid days from time series (default=F)
  
  ##.................................................
  # Visual report         
  timewindow = c("MM"), # window over which summary statistics are derived
  
  # !!! changed from TRUE to save computational time
  visualreport=TRUE) # report combined from g.part2 and g.part4 
  # !!!