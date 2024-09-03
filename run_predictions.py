import os, glob, time

root_path = "/ourdisk/hpc/ai2es/tornado/wofs-preds-2023-hgt/"
dates = ["20230510", "20230511", "20230512", "20230523", "20230524", "20230526"][:1]
done = False

try:
    for date in dates:
        path = root_path+date
        times = [i[-4:] for i in sorted(glob.glob(path+'/****', recursive=True))]
        for tm in times:
            job_count = 0
            for ens in range(1, 19):
                rel_path = f"{date}/{tm}/ENS_MEM_{ens}"
                if len(glob.glob(root_path.replace("hgt", "update")+rel_path+"/**.nc")) == 37:
                    continue
                else:
                    os.system(f"sbatch --array=1-37 --export=rel_path='{date}/{tm}/ENS_MEM_{ens}' lydia_scripts/wofs_raw_predictions_oscer.sh")
                    job_count += 1
                    if job_count > 0 and job_count % 40 == 0:
                        raise BreakException
except BreakException:
    pass
