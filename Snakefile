# configfile: "config.yaml"
import numpy as np
import os

# Variables
COIN_FLIP_BIAS = 0.5
OBJ_NAME = ['spokes', 'points_random', 'test_target'] # possible objects are: 'spokes', 'points_random', 'test_target'
NITER = 500
NA = 0.8
MAX_PHOTONS = 1e2
SEED=np.linspace(100,200,20).astype(int)

# COIN_FLIP_BIAS = np.linspace(1e-9,1-1e-10,20)
# NA = np.linspace(0.1,1.4,20)
# MAX_PHOTONS = np.logspace(0,4,20)

kwargs = {
    "coin_flip_bias":COIN_FLIP_BIAS,
    "niter":NITER,
    "na":NA,
    "max_photons":MAX_PHOTONS,
    "obj_name":OBJ_NAME,
    "seed":SEED  
}
 # Variables

base_dir = workflow.current_basedir
script = os.path.join(workflow.basedir,"simulate.py")
collate_script = os.path.join(workflow.basedir,"collate_csvs.py")

results = "{base_dir}/results/{coin_flip_bias}-{niter}-{na}-{max_photons}-{seed}-{obj_name}"

variables = [{"max_photons":np.logspace(0,4,200)},
            {"na":np.linspace(0.1,1.4,200)},
            {"coin_flip_bias":np.linspace(1e-9,1-1e-10,200)}]

results_full = []

for variable_dict in variables:
    results_expanded = expand(results+"/analyse_images.done",
                    base_dir = workflow.basedir,
                    **{**kwargs, **variable_dict}
                    )
    # breakpoint()
    results_full.extend(results_expanded)


# breakpoint()
container: "docker://snakemake/snakemake"

rule all:
    input:
        "results/full.csv"
rule generate_images:
    conda:
        "environment.yml"
    params:
        outdir=directory(results)
    resources:
        mem_mb=8000
    output:
        # out1=directory(results),
        touch(results+"/generate_images.done")
        # out3=touch("generate_images.done")
    shell:
        """
	    python {script} \
        --out_dir {params.outdir} \
        --coin_flip_bias {wildcards.coin_flip_bias} \
        --niter {wildcards.niter} \
        --na {wildcards.na} \
        --max_photons {wildcards.max_photons} \
        --obj_name {wildcards.obj_name} \
        --seed {wildcards.seed} \
        --no_show_figures \
        --no_save_images
        # --no_analysis
        """

rule analyse_images:
    input:
        results+"/generate_images.done"
    conda:
        "environment.yml"
    params:
        outdir=directory(results)
    resources:
        mem_mb=100
    output:
        touch(results+"/analyse_images.done")
    shell:
        """
	    # python {script} \
        # --out_dir {params.outdir} \
        # --coin_flip_bias {wildcards.coin_flip_bias} \
        # --niter {wildcards.niter} \
        # --na {wildcards.na} \
        # --max_photons {wildcards.max_photons} \
        # --obj_name {wildcards.obj_name} \
        # --seed {wildcards.seed} \
        # --no_show_figures \
        # --no_image_generation
        """

rule collate_csvs:
    input:
        # results+"/generate_images.done"
        results_full
    conda:
        "environment.yml"
    # params:
    #     outdir=directory(results)
    resources:
        mem_mb=1000
    output:
       "results/full.csv"
    shell:
        """
	    python {collate_script} \
        --out_dir results \
        """
