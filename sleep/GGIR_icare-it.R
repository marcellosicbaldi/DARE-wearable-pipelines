library(GGIR)

bronze_dir <- "/Users/augenpro/Documents/Age-IT/data/Bronze"
silver_dir <- "/Users/augenpro/Documents/Age-IT/data/Silver"
visit <- "T0 (baseline)"
sensors <- c("GeneActivPolso", "GeneActivCaviglia")

subjects <- sort(list.dirs(bronze_dir, full.names = FALSE, recursive = FALSE))
subjects <- c("08667", "20603", "36765")

for (sub in subjects) {
  
  print(sub)
  
  for (sensor in sensors) {
    
    print(sensor)
    
    datadir <- file.path(bronze_dir, sub, visit, sensor)
    # Find file that ends in bin (full path)
    bin_file_path <- list.files(datadir, pattern = "\\.bin$", full.names = TRUE)
    
    # Check if no bin file is found, skip this iteration
    if (length(bin_file_path) == 0) {
      cat("No .bin file found for", sub, "in", sensor, "\n")
      next # Going to the next iteration
    }
    
    # Output_dir (Silver layer)
    output_dir <- file.path(silver_dir, sub, visit, sensor)
    
    # Define different thresholds for wrist (Polso) and ankle (Caviglia)
    if (sensor == "GeneActiv Polso") { # Default GGIR values (average of multiple cut-points for wrist)
      threshold_lig <- c(40)
      threshold_mod <- c(100)
      threshold_vig <- c(400)
    } else if (sensor == "GeneActiv Caviglia") { # Bammann2021
      threshold_lig <- NULL   
      threshold_mod <- c(342)  
      threshold_vig <- NULL  
    }
  
    ###### GGIR CALL ######
    GGIR(
      
      datadir = as.character(bin_file_path),
      outputdir = output_dir,
      
      mode=c(1:5),
      
      do.report=c(2,4,5), # summary spreadsheet generation? 
      
      studyname = "icareit", # studyname must be specified if datadir is a list of files
      overwrite = FALSE, # overwrite analysis for which milestone data already exists?
      do.imp=FALSE, # imputing missing values by g.impute in GGIR g.part2?
      # !!! Changed from FALSE to save computational time
      do.parallel = TRUE, # Whether to use multi-core processing
      # !!!
      print.filename=FALSE, # Whether to print the filename before analysing it
      storefolderstructure = FALSE, # Whether storing folder structure of the acc data
      printsummary=TRUE, # Print summary of calibration procedure in the console when done?
      epochvalues2csv=TRUE, # reporting epoch values to a CSV file?
      minimum_MM_length.part5 = 12, # Min n.hours of MM day to be included in cleaned g.part5
      
      # !!! Specific for i.Careit
      idloc=2, # participant ID as the character string preceding "_" in the filename
      # !!!
      
      #.................................................
      # Part 2
      #.................................................
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
      mvpathreshold = NULL, # Numeric (default = NULL). Legacy parameter, if not provided GGIR uses the value of threshold.mod for this
      
      # !!! changed from TRUE to save computational time
      do.part2.pdf=FALSE, # Whether to generate a pdf for g.part2
      # !!!
      
      #.................................................
      # Part 3 + 4
      #.................................................
      timethreshold = c(5), # threshold (min) for sustained inactivity periods (default 5)
      anglethreshold = 5, # threshold (°) for sustained inactivity periods (default 5)
      ignorenonwear = TRUE, # If TRUE then ignore detected monitor non-wear periods
      
      outliers.only = FALSE, # if FALSE include all available nights in the visual pdf report
    
      # Part 4 parameters:
      #...........................
      excludefirstlast = FALSE, # ignore the first and last night
      
      # changed from 16 because of shorter recordings in our protocol
      includenightcrit = 10, # min no. of valid hours per night (24h window between noon and noon)
      
      def.noc.sleep = 1, # time window during which sustained inactivity will be assumed to represent sleep (default=1) - only used if no sleep log entry is available
      
      loglocation = NULL, # Path to sleep log csv (default NULL --> if NULL uses HDCZA)

      relyonguider = TRUE, # whether SIB that overlap with the guider are labelled as sleep
      
      sleepwindowType = "SPT", # since diary reported lights off 
      HASPT.algo = "HDCZA", # Do only automatic
      HASIB.algo = "vanHees2015", # default SIB detection,
      
      #.................................................
      # Part 5
      #.................................................
      threshold.lig = c(40), # Threshold for light PA to separate from inactivity (def 40)
      threshold.mod = c(100), # Threshold for moderate PA to separate from light (def 100)
      threshold.vig = c(400), # Threshold for vigorous PA to separate from moderate (def 400)
      boutcriter = 0.8, # what fraction of a bout should be above the mvpathreshold (def .8)
      boutcriter.in = 0.9, # fraction of a bout should be below the threshold.lig (def .9)
      boutcriter.lig = 0.8, # fraction that should be between threshold.lig and threshold.mod (def 0.8)
      boutcriter.mvpa = 0.8, # fraction of a bout needs to be above the threshold.mod (def c(10,20,30))
      boutdur.in = c(10,20,30), # duration(s) (min) of inactivity bouts to be extracted (def c(10, 20, 30))
      boutdur.lig = c(1,5,10), # duration(s) (min) of light activity bouts to be extracted (def c(1,5,10))
      boutdur.mvpa = c(1,5,10), # Duration(s) of MVPA bouts in minutes (def c(1, 5, 10))
      includedaycrit.part5 = 2/3, # Inclusion criteria for no. of valid waking hours (def 2/3)
      frag.metrics="all", # Fragmentation metric to extract
      save_ms5rawlevels = TRUE, # Save the time series classification (levels)
      save_ms5raw_format = "csv", # how data should be stored (csv or RData)
      part5_agg2_60seconds=TRUE, # use aggregate epochs to 60 sec as part of the g.part5 analysis
      save_ms5raw_without_invalid=FALSE, # remove invalid days from time series (default=F)
      
      ##.................................................
      # Visual report         
      timewindow = c("MM"), # window over which summary statistics are derived
      
      visualreport=TRUE, # report combined from g.part2 and g.part4 
      
      #.................................................
      # Part 6
      #.................................................
      # For now using the default parameters, but has to be investigated
    )
  }
}



# Try to embed an external function to save to parquet