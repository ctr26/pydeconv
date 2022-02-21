# configfile: "config.yaml"
import numpy as np
import os

# Variables
COIN_FLIP_BIAS = 0.5
OBJ_NAME = 'spokes' # possible objects are: 'spokes', 'points_random', 'test_target'
NITER = 200
NA = 0.8
MAX_PHOTONS = 1e+2

results = "results/{coin_flip_bias}-{niter}-{na}-{max_photons}-{obj_name}"

all_results = expand(results,
            base_dir = workflow.basedir,
            coin_flip_bias=COIN_FLIP_BIAS,
            niter=NITER,
            na=NA,
            max_photons=MAX_PHOTONS,
            obj_name=OBJ_NAME
            )

# rule all:
#     input:
#         "out/{psf_type}_{psf_width}_{signal_strength}.csv"
# expand("out/{psf_type}_{psf_width}_{signal_strength}.csv",psf_type=PSF_TYPE,PSF_WIDTH,SIGNAL_STRENGTH)

base_dir = workflow.current_basedir
script = os.path.join(workflow.basedir,"simulate.py")

rule all:
    input:
        all_results
        # "out/{psf_type}_{psf_width}_{signal_strength}.csv"


rule generate_images:
    # input:
    #     "{basedir}/040520_pres_cluster_coins.py"
    conda:
        "environment.yml"
    params:
        # shell=os.environ["SHELL"]
        # psf_scale="{psf_scale}"
        # psf_type = "{psf_type}",
        # psf_scale = "{psf_scale}",
        # signal_strength = "{signal_strength}",
        # thinning_type = "{thinning_type}"
    resources:
        mem_mb=4000
    output:
        directory(results)
    shell:
        """
	    python {script} \
        --out_dir {output} \
        --coin_flip_bias {wildcards.coin_flip_bias} \
        --niter {wildcards.niter} \
        --na {wildcards.na} \
        --max_photons {wildcards.max_photons} \
        --obj_name {wildcards.obj_name} \
        --no_analysis
        """

rule analyse_images:
    input:
        results
    conda:
        "environment.yml"
    params:
        # shell=os.environ["SHELL"]
        # psf_scale="{psf_scale}"
        # psf_type = "{psf_type}",
        # psf_scale = "{psf_scale}",
        # signal_strength = "{signal_strength}",
        # thinning_type = "{thinning_type}"
    resources:
        mem_mb=4000
    output:
        directory(results)
    shell:
        """
	    python {script} \
        --out_dir {output} \
        --coin_flip_bias {wildcards.coin_flip_bias} \
        --niter {wildcards.niter} \
        --na {wildcards.na} \
        --max_photons {wildcards.max_photons} \
        --obj_name {wildcards.obj_name} \
        --no_image_generation
        """


# parser.add_argument("--signal_strength", default=signal_strength, type=float)
# parser.add_argument("--coin_flip_bias", default=coin_flip_bias, type=float)
# parser.add_argument("--savefig", default=savefig, type=int)
# parser.add_argument("--save_images", default=save_images, type=int)
# parser.add_argument("--image_scale", default=image_scale, type=int)
# parser.add_argument("--psf_scale", default=psf_scale, type=float)
# parser.add_argument("--psf_gradient", default=psf_gradient, type=float)
# parser.add_argument("--psf_type", default=psf_type, type=str)
# parser.add_argument("--max_iter", default=max_iter, type=int)
# parser.add_argument("--thinning_type", default=thinning_type, type=str)
# parser.add_argument("--out_dir", default=out_dir, type=str)
# parser.add_argument("--background_L", default=background_L, type=float)
# parser.add_argument("--background_k", default=background_k, type=float)