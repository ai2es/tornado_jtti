import os, glob

root_path = "/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-hgt/"
dates = ["20230510", "20230511", "20230512", "20230523", "20230524", "20230526"]

for date in dates:
    dir = root_path+date
    times = [i[-4:] for i in sorted(glob.glob(dir+'/****', recursive=True))]
    for time in times:
        for ens in range(1, 19):
            rel_path = f"{date}/{time}/ENS_MEM_{ens}"
            if len(glob.glob(root_path+rel_path+"/**.nc")) == 37:
                continue
            else:
                os.system(f"sbatch --array=1-37 --export=rel_path='{date}/{time}/ENS_MEM_{ens}' lydia_scripts/wofs_raw_predictions_oscer.sh")
