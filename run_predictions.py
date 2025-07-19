import os

dates = ["20230510", "20230511", "20230512", "20230523", "20230524", "20230526"][:1]
for date in dates:
    os.system(f"sbatch --array=1-21 --export=date='20230512' wofs_raw_predictions_oscer.sh")
