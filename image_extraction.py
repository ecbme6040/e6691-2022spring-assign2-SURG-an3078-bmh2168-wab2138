import os

import multiprocessing
import workers
import numpy as np
def transform_all(data_path="./e6691-bucket-videos/surgery.videos.hernia/"):
    
    directory = os.fsencode(data_path)
    njobs=2
    with multiprocessing.Pool(4) as p:
        results=p.map(workers.process_vid, os.listdir(directory))
    return results
transform_all()