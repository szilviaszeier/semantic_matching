import glob
import logging
import os

from utils.tools import annotate_dataset, setup_logging_config

if __name__ == '__main__':
    setup_logging_config()

    sample = 1
    annotations_file_name = f"annotations_{sample}.json"
    dataset_dir = "./semantic_matching"
    selected_dirs = ["supix", "transf", "seman"]

    environments_name = glob.glob(f"{dataset_dir}/*")
    for env_name in environments_name:
        env_states = glob.glob(f"{env_name}/*")
        for env_state in env_states:
            env_views = glob.glob(f"{env_state}/*")
            for segms in env_views[:]:
                if segms.split("/")[-1] in selected_dirs:
                    logging.info(f"Annotating directory named: {segms}")

                    img_dir = os.path.join("/".join(segms.split("/")[:-1]), "image")
                    annotations = annotate_dataset(segms, img_dir, annotations_file_name, sample=sample)
