# configfile: "config.yaml"
import numpy as np
import os

# Variables
COIN_FLIP_BIAS = np.linspace(0.5,1,10)
OBJ_NAME = ['spokes', 'points_random', 'test_target'] # possible objects are: 'spokes', 'points_random', 'test_target'
NITER = 200
NA = np.linspace(0.1,2,10)
MAX_PHOTONS = np.logspace(0,6,10)
SEED=10

base_dir = workflow.current_basedir
script = os.path.join(workflow.basedir,"simulate.py")

results = "{base_dir}/results/{coin_flip_bias}-{niter}-{na}-{max_photons}-{seed}-{obj_name}"


container: "docker://continuumio/miniconda3:4.4.10"

rule all:
    input:
        all_results = expand(results+"/analyse_images.done",
                base_dir = workflow.basedir,
                coin_flip_bias=COIN_FLIP_BIAS,
                niter=NITER,
                na=NA,
                max_photons=MAX_PHOTONS,
                obj_name=OBJ_NAME,
                seed=SEED
                )
        # all_results = expand(results+"/analyse_images.done",
        #             base_dir = workflow.basedir,
        #             coin_flip_bias=0.5,
        #             niter=200,
        #             na=0.8,
        #             max_photons=1e+2,
        #             obj_name='spokes',
        #             seed=10
        #             )

rule generate_images:
    conda:
        "environment.yml"
    params:
        outdir=directory(results)
    resources:
        mem_mb=2000
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
        mem_mb=1000
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
