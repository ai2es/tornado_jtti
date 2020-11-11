
1. Before storm-tracking, you don't have to do any processing, except make sure that GridRad files are in the directory 
structure shown in the attached image.  This ensures that they can be found by `gridrad_io.find_file`, the method found here.

2. Before storm-tracking, I suggest that you run echo classification.  This filters out non-convective pixels, so that 
storm-tracking considers only convective pixels.  Echo classification is done with a modified version of Cameron 
Homeyer's SL3D algorithm (Storm-labeling in 3 Dimensions).  The "modified version" was created by me, and the main 
differences are
    1. mine is Python code, whereas Cameron's is IDL code; 
    2. mine focuses only on classifying convective vs. non-convective, while Cameron's has other categories, like anvil 
and stratiform and transition region.  
    3. To run echo classification, you can use the script here.  There's a lot of hyperparameters 
    (peakedness neighbourhood, max peakedness height, minimum echo-top height, etc.), which I suggest leaving at 
    their default values.  You can read more about the echo-classification algorithm in Section 3.1.2 of my dissertation (here).

3. Storm-tracking is done in two steps: preliminary and final (analogous to segmotion and besttrack in WDSS-II).  
To do preliminary tracking, run the script run_echo_top_tracking.py independently on each day.  There's a lot of 
hyperparameters (reflectivity cutoff for echo-top height, minimum echo-top height, minimum storm size, 
minimum interstorm distance, etc.), which I suggest leaving at their default values.  After preliminary tracking, 
you need to run final tracking on each set of consecutive days.  For example, say the first few days of your dataset 
are: Jan 1 2020, Jan 2 2020, Jan 3 2020, Jan 6 2020, Jan 7 2020.  You would run final tracking on 
Jan 1-3 2020 and Jan 6-7 2020 independently.  There would be no sense in running final tracking on Jan 1-7 2020 
(although you could), because there's a two-day gap and obviously no storm lasts that long.  Final tracking is done 
with the script reanalyze_storm_tracks.py.  Again, I suggest leaving all hyperparameters at their default values.  
Note that run_echo_top_tracking.py, which does the preliminary tracking, also does storm detection 
(finding out where storms are at each time step, before connecting the tracks over time).  You can read about my 
storm-detection and -tracking algorithms in Sections 3.1.3 and 3.1.4 of my dissertation.

4. Matching with tornado reports is also done in two steps: preliminary and final.  To do preliminary matching, 
run the script link_tornadoes_to_storms.py independently on each day.  Make sure that genesis_only = 0.  
If you make genesis_only = 1, the script will link storms only to tornadogenesis events, so the label will end up 
being "does this storm produce a *new* tornado in the next K minutes?"  If you make genesis_only = 0, 
the script will link storms to tornado tracks in general, so the label will end up being "is this storm tornadic at 
any time in the next K minutes (whether or not the tornado already exists)?"  The first question (predicting new 
tornadoes only) caused ML to do weird things, because the ML had to predict "no" for storms that were already tornadic 
and didn't produce a new tornado.  After preliminary linkage, you need to do final linkage, which processes multiple 
days at a time (so that you don't have a cutoff every 24 hours where linkages get dropped).  
This is done with the script share_linkages_across_spc_dates.py.

5. After matching with tornado reports, you need to create labels (correct answers, which are a "yes" or "no" for 
each storm object), using the script compute_target_values.py.  You can ignore the input arguments that deal with "wind_speed".  These are only if you're predicting straight-line wind, not tornadoes.

6. To create storm-centered radar images (I think DJ calls these "storm patches"), which are used as predictors, use the script extract_storm_images_from_gridrad.py.  I suggest making rotate_grids = 1 (so that storm motion is always towards the right, or in the positive x-direction).  Also, you'll need to install the library srtm (https://github.com/tkrajina/srtm.py).  It's not available on pip or conda, so you'll have to do a "git clone" and "python setup.py install", but the installation is quick and smooth (should take you a minute with zero complications).  srtm deals with surface elevation, which is needed because in the storm-centered images created by extract_storm_images_from_gridrad.py, the height coordinate is ground-relative instead of sea-level-relative.

7. Once you have storm-centered radar images (predictors) and labels (predictands), you can train the CNN.  Honestly, 
the training code is messy and you would probably be better off writing your own.  However, if you would like me to share the training code (architecture, generator, and training scripts), I can do that in a separate email.  Also, note that I used two data sources for predictors: storm-centered radar images and proximity soundings from the Rapid Refresh (RAP) model.  If you also want to use proximity soundings, let me know and I'll send the code.  I'm holding off on that for now, because (a) this email is long enough and (b) dealing with RAP data is cumbersome.  The archive is incomplete, and you have to deal with data on different grids.  At some times the 13-km grid is available, but at some times only the 20-km grid is available, so the code has conditional statements to deal with this.  Also, dealing with RAP data means reading grib files into Python, which is an extra can of worms.  But with that said, I think including soundings (or any near-storm information) is worth it.