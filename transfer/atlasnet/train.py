"""
Main training script.
Author: Thibault Groueix 01.11.2019
Modified on October 12th, 2020
@modifier: urikotlicki
"""

# import system modules
import os.path as osp
import sys
import time
import torch

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
import transfer.atlasnet.auxiliary.argument_parser as argument_parser
import transfer.atlasnet.auxiliary.my_utils as my_utils
import transfer.atlasnet.training.trainer as trainer
from transfer.atlasnet.auxiliary.my_utils import yellow_print

opt = argument_parser.parser()
torch.cuda.set_device(opt.multi_gpu[0])
my_utils.plant_seeds(random_seed=opt.random_seed)

trainer = trainer.Trainer(opt)
trainer.build_dataset()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()
trainer.start_train_time = time.time()

if opt.mode == 'test' and opt.custom_data:
    with torch.no_grad():
        trainer.test_epoch()
        sys.exit(0)
else:
    if opt.demo:
        with torch.no_grad():
            trainer.demo(opt.demo_input_path)
        sys.exit(0)

    if opt.run_single_eval:
        with torch.no_grad():
            trainer.test_epoch()
        sys.exit(0)

    for epoch in range(trainer.epoch, opt.nepoch):
        trainer.train_epoch()
        with torch.no_grad():
            trainer.test_epoch()
        trainer.dump_stats()
        trainer.increment_epoch()
        trainer.save_network()

    if not opt.custom_data:
        yellow_print(f"Visdom url http://localhost:{trainer.opt.visdom_port}/")
        yellow_print(f"Netvision report url http://localhost:{trainer.opt.http_port}/{trainer.opt.dir_name}/index.html")

    yellow_print(f"Training time {(time.time() - trainer.start_time)//60} minutes.")
