import time

from utils import *


def train_epoch(epoch, model, ema_model, train_loader,
                optimizer, criterion, batch_logger, epoch_logger, args):
    """a epoch of training

    :param epoch: begin from 0
    :param model: model in device
    :param ema_model: ema model in device
    :param train_loader:
    :param optimizer:
    :param criterion: a dict of keys 'classification' and 'consistency'
    :param batch_logger:
    :param epoch_logger:
    :param args:
    :return:
    """
    print('Train at epoch: {}'.format(epoch + 1))

    time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    cross_entropy_meter = AverageMeter()
    consistency_meter = AverageMeter()
    ema_ce_meter = AverageMeter()
    ema_cons_meter = AverageMeter()
    weight_meter = AverageMeter()
    top1 = AverageMeter()
    ema_top1 = AverageMeter()
    lr_meter = AverageMeter()

    model.train()
    ema_model.train()

    end_time = time.time()
    for step, ((student_input, teacher_input), target) in enumerate(train_loader):
        time_meter.update(time.time() - end_time)
        adjust_learning_rate(optimizer, epoch, step, len(train_loader), args)
        lr_meter.update(optimizer.param_groups[0]['lr'])

        student_input = student_input.cuda()
        teacher_input = teacher_input.cuda()
        target = target.cuda()

        labeled_batch_size = target.detach().ne(args.NO_LABEL).sum()
        minibatch_size = target.size(0)
        # print('labeled batch size {}/{}'.format(labeled_batch_size, minibatch_size))

        out = model(student_input)
        ema_out = ema_model(teacher_input)

        class_loss = criterion['classification'](out, target) / minibatch_size
        cross_entropy_meter.update(class_loss.item(), minibatch_size)

        # check teacher classification loss
        ema_class_loss = criterion['classification'](ema_out, target) / minibatch_size
        ema_ce_meter.update(ema_class_loss.item(), minibatch_size)

        # calculate consistency loss
        consistency_weight = get_current_consistency_weight(epoch,
                                                            args.consistency_weight,
                                                            args.rampup_length)
        weight_meter.update(consistency_weight)
        consistency_loss = criterion['consistency'](out, ema_out) * consistency_weight / minibatch_size
        consistency_meter.update(consistency_loss.item(), minibatch_size)

        loss = class_loss + consistency_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss diverge {}'.format(loss.item())
        loss_meter.update(loss.item(), minibatch_size)

        prec = accuracy(out.detach(), target.detach(), args, topk=(1,))  # here outcome is a list
        top1.update(prec[0].item(), minibatch_size)

        ema_prec = accuracy(ema_out.detach(), target.detach(), args, topk=(1,))
        ema_top1.update(ema_prec[0].item(), minibatch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step = epoch * len(train_loader) + step  # step begins also from 0
        update_ema_variable(model, ema_model, args.ema_decay, global_step)

        batch_time_meter.update(time.time() - end_time)
        end_time = time.time()

        # batch_logger: epoch, step, global_step, class_loss, cons_loss, prec1, ema_prec1, lr
        batch_logger.log({'epoch': epoch + 1, 'batch': step + 1, 'iter': global_step + 1,
                          'class_loss': cross_entropy_meter.val,
                          'consistency_loss': consistency_meter.val,
                          'prec1': top1.val, 'ema_prec1': ema_top1.val,
                          'lr': optimizer.param_groups[0]['lr']})
        # print(batch_time_meter.val)
        # print(time_meter.val)
        # print(cross_entropy_meter.val)
        # print(consistency_meter.val)
        # print(top1.val)
        # print(ema_top1.val)
        # print(lr_meter.val)

        if (step + 1) % 10 == 0:
            # import pdb;pdb.set_trace()
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Class_loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t'
                  'Cons_loss {cons_loss.val:.4f} ({cons_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Ema_Prec@1 {ema_top1.val:.5f} ({ema_top1.avg:.5f})'.format(epoch + 1,
                                                                          step + 1,
                                                                          len(train_loader),
                                                                          batch_time=batch_time_meter,
                                                                          data_time=time_meter,
                                                                          c_loss=cross_entropy_meter,
                                                                          cons_loss=consistency_meter,
                                                                          top1=top1,
                                                                          ema_top1=ema_top1,
                                                                          lr=lr_meter.val))

    epoch_logger.log({'epoch': epoch + 1, 'class_loss': cross_entropy_meter.val,
                      'consistency_loss': consistency_meter.val, 'prec1': top1.val,
                      'ema_prec1': ema_top1.val})


def validate_epoch(epoch, model, val_loader, criterion, val_logger, args):
    print('Validation at epoch {}'.format(epoch + 1))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        minibatch_size = target.size(0)
        labeled_minibatch_size = target.detach().ne(args.NO_LABEL).sum()

        output = model(input)

        class_loss = criterion['classification'](output, target) / minibatch_size
        losses.update(class_loss.item(), minibatch_size)

        prec = accuracy(output, target, args, topk=(1,))
        top1.update(prec[0].item(), minibatch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'.format(epoch + 1,
                                                                    i + 1,
                                                                    len(val_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time,
                                                                    loss=losses,
                                                                    top1=top1))
    val_logger.log({'epoch': epoch + 1, 'loss': losses.avg, 'prec1': top1.avg})

    return losses.avg, top1.avg
