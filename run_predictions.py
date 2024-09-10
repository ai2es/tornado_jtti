import os, glob, time

root_path = "/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-hgt/"
dates = ["20230510", "20230511", "20230512", "20230523", "20230524", "20230526"][:1]
done = False

try:
    for date in dates:
        path = root_path+date
        times = [i[-4:] for i in sorted(glob.glob(path+'/****', recursive=True))]
        for tm in times:
            rel_path = f"{date}/{tm}"
            os.system(f"sbatch --array=1-666 --export=rel_path='{date}/{tm}' lydia_scripts/wofs_raw_predictions_oscer.sh")