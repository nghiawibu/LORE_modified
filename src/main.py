from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, load_str, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from models.classifier import Processor
from str_model.models import Model
from str_model.utils import get_args, TokenLabelConverter
from str_model.demo import run_model, load_img
import string

def main(opt, str_opt, device):
  torch.autograd.set_detect_anomaly(True)
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  #print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  Trainer = train_factory[opt.task]
 

  processor = Processor(opt)
  processor.train()
  optimizer = torch.optim.Adam([  \
              {'params': model.parameters()}, \
              {'params': processor.parameters()}],  lr =opt.lr, betas= (0.9, 0.98), eps=1e-9)
  if str_opt.sensitive:
    str_opt.character = string.printable[:-6]
  # cudnn.benchmark = True
  # cudnn.deterministic = True
  str_opt.num_gpu = torch.cuda.device_count()
  
  str_opt.saved_model = str_opt.model_dir
  converter = TokenLabelConverter(str_opt)
  str_opt.num_class = len(converter.character)
  
  if str_opt.rgb:
      str_opt.input_channel = 3
  str_model = Model(str_opt)
  trainer = Trainer(opt, model, optimizer, processor, str_model, str_opt, converter)



  start_epoch = 0
  if opt.load_model != '':
    #model, optimizer, start_epoch = load_model(model, opt.load_model)
      #model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    model = load_model(model, opt.load_model)

  if opt.load_processor != '':
    processor = load_model(processor, opt.load_processor)

  str_model = load_str(str_model, str_opt.saved_model, device)
  
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)

    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
   
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'processor_{}.pth'.format(mark)), 
                epoch, processor, optimizer)
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))

      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'processor_best.pth'), 
                  epoch, processor)
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                  epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'processor_last.pth'), 
                epoch, processor, optimizer)
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'processor_{}.pth'.format(epoch)), 
                epoch, processor, optimizer)
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  str_opt = get_args(is_train=False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  main(opt, str_opt, device)
