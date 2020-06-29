import argparse
import cv2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from pathlib import Path
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--conf-threshold", default=0.6, type=float)
    args = parser.parse_args()
    
    return args

def main(args):
    # Get the configuration ready
    logger = setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.conf_threshold
    dicts = list(DatasetCatalog.get("bulb_wise_tl_train"))
    metadata = MetadataCatalog.get("bulb_wise_tl_train")

    predictor = DefaultPredictor(cfg)
    inputs = Path(args.input)
    if inputs.is_file():
        im = cv2.imread(str(inputs))
        tic = time.time()
        outputs = predictor(im)
        toc = time.time()
        logger.info("Time consumed per frame: {}.".format(toc-tic))
        instances = outputs['instances'].to('cpu')
        scores = np.asarray([x for x in instances.scores])
        chosen = (scores > args.conf_threshold).nonzero()[0]
        instances = instances[chosen]
        labels = instances.pred_classes.detach().numpy()
        instances.pred_classes = labels
        v = Visualizer(im[:,:,::-1], metadata)
        v = v.draw_instance_predictions(instances)
        img = v.get_image()[:, :, ::-1]
        cv2.imshow('Test on {}.'.format(str(inputs)), img)
        cv2.waitKey(0)
    else:
        img_lists = list(inputs.glob('*.jpg'))
        for img in img_lists:
            im = cv2.imread(str(img))
            tic = time.time()
            outputs = predictor(im)
            toc = time.time()
            logger.info("Time consumed per frame: {}.".format(toc - tic))
            instances = outputs['instances'].to('cpu')
            scores = np.asarray([x for x in instances.scores])
            chosen = (scores > args.conf_threshold).nonzero()[0]
            instances = instances[chosen]
            labels = instances.pred_classes.detach().numpy()
            instances.pred_classes = labels
            v = Visualizer(im[:,:,::-1], metadata)
            v = v.draw_instance_predictions(instances)
            img = v.get_image()[:, :, ::-1]
            cv2.imshow('Test on {}.'.format(str(inputs)), img)
            cv2.waitKey(0)

if __name__ == "__main__":
    args = parse_args()
    main(args)