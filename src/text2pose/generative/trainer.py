import os
import shutil
import time
import numpy as np
from tqdm import tqdm, trange
import logging
import torch 
import roma
import math
import json


# import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
from text2pose.vocab import Vocabulary # needed
from text2pose.data import PoseScript
from text2pose.utils import save_to_file, load_from_file
from text2pose.generative.fid import FID
from text2pose.generative.our_model_generative import CondTextPoser
# from text2pose.generative.model_generative import CondTextPoser

logger = logging.getLogger(__name__)

def laplacian_nll(x_tilde, x, log_sigma):
    log_norm = - (np.log(2) + log_sigma)
    log_energy = - (torch.abs(x_tilde - x)) / torch.exp(log_sigma)
    return - (log_norm + log_energy)


def gaussian_nll(x_tilde, x, log_sigma):
    log_norm = - 0.5 * (np.log(2 * np.pi) + log_sigma)
    log_energy = - 0.5 * F.mse_loss(x_tilde, x, reduction='none') / torch.exp(log_sigma)
    return - (log_norm + log_energy)


def load_model(model_path, device):
	
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cuda')
	text_encoder_name = ckpt['args'].text_encoder_name
	latentD = ckpt['args'].latentD
	
	# load model
	model = CondTextPoser(text_encoder_name=text_encoder_name, latentD=latentD).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print("Loaded model from:", model_path)
	
	return model, text_encoder_name


def compute_eval_metrics(model, dataset, fid_version, device):

	# NOTE: fid_version should be of the format (retrieval_model_shortname, seed)

	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=1,
		num_workers=8,
		pin_memory=True,
	)

	results = {}

	# compute FID
	fid = FID(version=fid_version, device=device)
	fid.extract_real_features(data_loader)
	fid.reset_gen_features()
	for batch in tqdm(data_loader):
		caption_tokens = batch['caption_tokens'].to(device)
		caption_lengths = batch['caption_lengths'].to(device)
		caption_tokens = caption_tokens[:,:caption_lengths.max()]
		with torch.inference_mode():
			onepose = model.sample_text_nposes(caption_tokens, caption_lengths, n=1)['pose_body']
		fid.add_gen_features( onepose )
	fid_value = fid.compute()
	results["fid"] = fid_value

	# compute elbos
	body_model = BodyModel(bm_fname = config.SMPLH_NEUTRAL_BM, num_betas = config.n_betas).to(device)
	elbos = {'v2v': 0.0, 'jts': 0.0, 'rot': 0.0}
	for batch in tqdm(data_loader):
		poses = batch['pose'].to(device)
		caption_tokens = batch['caption_tokens'].to(device)
		caption_lengths = batch['caption_lengths'].to(device)
		caption_tokens = caption_tokens[:,:caption_lengths.max()]
		with torch.inference_mode():
			output = model.forward(poses, caption_tokens, caption_lengths)
			bm_rec = body_model(pose_body=output['pose_body_pose'][:,1:22].flatten(1,2),
								pose_hand=output['pose_body_pose'][:,22:].flatten(1,2),
								root_orient=output['pose_body_pose'][:,:1].flatten(1,2))
			bm_orig = body_model(pose_body=poses[:,1:22].flatten(1,2),
								pose_hand=poses[:,22:].flatten(1,2),
								root_orient=poses[:,:1].flatten(1,2))
			kld = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output['t_z']), dim=[1]))
		elbos['v2v'] += (-laplacian_nll(bm_orig.v, bm_rec.v, model.decsigma_v2v).sum() - kld).detach().item()
		elbos['jts'] += (-laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, model.decsigma_jts).sum() - kld).detach().item()
		elbos['rot'] += (-gaussian_nll(output['pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), model.decsigma_rot).sum() - kld).detach().item()
	
	for k in elbos: elbos[k] /= len(dataset)
	results.update(elbos)

	# normalize results
	norm_results = {'fid':results['fid'],
					'jts':results['jts']/(len(bm_orig.Jtr[0]) * 3),
					'v2v':results['v2v']/(len(bm_orig.v[0]) * 3),
					'rot':results['rot']/(model.pose_decoder.num_joints * 9)}

	return norm_results

def get_seed_from_model_path(model_path):
    return model_path.split("/")[-2][len("seed"):]


def get_epoch_from_model_path(model_path):
	return model_path.split("_")[-1].split(".")[0]


class Trainer():
    def __init__(self, args):
        self.args = args

        if self.args.mode == 'train':
            self.model = CondTextPoser(text_encoder_name=args.text_encoder_name, latentD=args.latentD)
            self.body_model = BodyModel(bm_fname = config.SMPLH_NEUTRAL_BM, num_betas = config.n_betas)
            self.device = "cuda:0"
            self.model.to(self.device)
            # self.body_model = body_model.to(self.device)
            self.body_model.to(self.device)


    def train(self):
        self.body_model.eval()
        ckpt_fname = os.path.join(self.args.output_dir, 'checkpoint_last.pth')
        print('Load training dataset')
        dataset_train = PoseScript(version=self.args.dataset, split='train', text_encoder_name=self.args.text_encoder_name, caption_index='rand')
        print(dataset_train, len(dataset_train))
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train, sampler=None, shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print('Load validation dataset')
        dataset_val = PoseScript(version=self.args.dataset, split='val', text_encoder_name=self.args.text_encoder_name, caption_index="deterministic-mix")
        print(dataset_val, len(dataset_val))
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, sampler=None, shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        # print(dataset_val, len(dataset_val))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        fid = FID((self.args.fid, self.args.seed), device=self.device)
        fid.extract_real_features(val_dataloader)


        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                         step_size=self.args.lr_step,
        #                                         gamma=self.args.lr_gamma,
        #                                         last_epoch=-1)



        train_iterator = trange(int(self.args.epochs), desc="Epoch")
        global_step = 0
        global_loss = 0
        start_time = time.time()
        best_score = None
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            self.model.train()
            for step, batch in enumerate(epoch_iterator):
                
                poses = batch['pose'].to(self.device)
                caption_tokens = batch['caption_tokens'].to(self.device)
                caption_lengths = batch['caption_lengths'].to(self.device)
                caption_tokens = caption_tokens[:,:caption_lengths.max()].to(self.device) # truncate within the batch, based on the longest text 
                
                with torch.set_grad_enabled(True):
                    output = self.model(poses, caption_tokens, caption_lengths)
                    bm_rec = self.body_model(pose_body=output['pose_body_pose'][:,1:22].flatten(1,2),
                                        pose_hand=output['pose_body_pose'][:,22:].flatten(1,2),
                                        root_orient=output['pose_body_pose'][:,:1].flatten(1,2))
                with torch.no_grad():
                    bm_orig = self.body_model(pose_body=poses[:,1:22].flatten(1,2),
                                        pose_hand=poses[:,22:].flatten(1,2),
                                        root_orient=poses[:,:1].flatten(1,2))
                    
                # compute losses 
                losses = {}
                
                # -- reconstruction losses
                losses[f'v2v'] = torch.mean(laplacian_nll(bm_orig.v, bm_rec.v, self.model.decsigma_v2v))
                losses[f'jts'] = torch.mean(laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, self.model.decsigma_jts))
                losses[f'rot'] = torch.mean(gaussian_nll(output[f'pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), self.model.decsigma_rot))

                # -- KL losses
                losses['kldpt'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output['t_z']), dim=[1]))
                
                # -- KL regularization losses
                bs = poses.size(0)
                n_z = torch.distributions.normal.Normal(
                    loc=torch.zeros((bs, self.model.latentD), device=self.device, requires_grad=False),
                    scale=torch.ones((bs, self.model.latentD), device=self.device, requires_grad=False))
                losses['kldnp'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], n_z), dim=[1])) if self.args.wloss_kldnpmul else torch.tensor(0.0)
                losses['kldnt'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['t_z'], n_z), dim=[1])) if self.args.wloss_kldntmul else torch.tensor(0.0)
                    
                # -- total loss
                loss = losses['v2v'] * self.args.wloss_v2v + \
                    losses['jts'] * self.args.wloss_jts + \
                    losses['rot'] * self.args.wloss_rot + \
                    losses['kldpt'] * self.args.wloss_kld + \
                    losses['kldnp'] * self.args.wloss_kldnpmul * self.args.wloss_kld + \
                    losses['kldnt'] * self.args.wloss_kldntmul * self.args.wloss_kld
                loss_value = loss.item()

                global_loss += loss_value
                global_step +=1

                
                # elbos (normalization is a bit different than for the losses)
                elbos = {}
                elbos[f'v2v'] = (-laplacian_nll(bm_orig.v, bm_rec.v, self.model.decsigma_v2v).sum()/2./bs - losses['kldpt']).detach().item()
                elbos[f'jts'] = (-laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, self.model.decsigma_jts).sum()/2./bs - losses['kldpt']).detach().item()
                elbos[f'rot'] = (-gaussian_nll(output[f'pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), self.model.decsigma_rot).sum()/2./bs - losses['kldpt']).detach().item()
                
            
                # step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



                ##LOGGER##
            logger.info("***** Running training *****")
            logger.info("Epoch: %d, loss: %d", epoch, loss_value)
            for key in (elbos.keys()):
                logger.info("  %s = %s", key, str(elbos[key]))  
            for key in (losses.keys()):
                logger.info("  %s = %s", key, str(losses[key]))  
            
            train_results = {k: v.item() for k, v in losses.items()}
            # train_results.update(losses)
            train_results.update(elbos)

            val_results = self.validate(val_dataloader, fid)
            logger.info("***** Validation *****")
            for key in (val_results.keys()):
                logger.info("  %s = %s", key, str(val_results[key]))  

                # save checkpoint
            tosave = {'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': self.args}
            # ... current checkpoint
            torch.save(tosave, ckpt_fname)

            log_stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]["lr"]}
            log_stats.update(val_results)
            log_stats.update(train_results)
            with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        logger.info("**********Training Finished***********")
        logger.info("Training Time: %d", time.time() - start_time)
        # return global_loss / global_step


    def validate(self, dataloader, fid):    

        global_step = 0
        global_loss = 0
        start_time = time.time()
        best_score = None

        fid.reset_gen_features()
        self.model.train()
        epoch_iterator = tqdm(dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):


            poses = batch['pose'].to(self.device)
            caption_tokens = batch['caption_tokens'].to(self.device)
            caption_lengths = batch['caption_lengths'].to(self.device)
            caption_tokens = caption_tokens[:,:caption_lengths.max()].to(self.device) # truncate within the batch, based on the longest text 
            
            with torch.set_grad_enabled(False):
                output = self.model(poses, caption_tokens, caption_lengths)
                bm_rec = self.body_model(pose_body=output['pose_body_pose'][:,1:22].flatten(1,2),
                                    pose_hand=output['pose_body_pose'][:,22:].flatten(1,2),
                                    root_orient=output['pose_body_pose'][:,:1].flatten(1,2))
            with torch.no_grad():
                bm_orig = self.body_model(pose_body=poses[:,1:22].flatten(1,2),
                                    pose_hand=poses[:,22:].flatten(1,2),
                                    root_orient=poses[:,:1].flatten(1,2))
                
            # compute losses 
            losses = {}
            
            # -- reconstruction losses
            losses[f'v2v'] = torch.mean(laplacian_nll(bm_orig.v, bm_rec.v, self.model.decsigma_v2v))
            losses[f'jts'] = torch.mean(laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, self.model.decsigma_jts))
            losses[f'rot'] = torch.mean(gaussian_nll(output[f'pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), self.model.decsigma_rot))

            # -- KL losses
            losses['kldpt'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output['t_z']), dim=[1]))
            
            # -- KL regularization losses
            bs = poses.size(0)
            n_z = torch.distributions.normal.Normal(
                loc=torch.zeros((bs, self.model.latentD), device=self.device, requires_grad=False),
                scale=torch.ones((bs, self.model.latentD), device=self.device, requires_grad=False))
            losses['kldnp'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], n_z), dim=[1])) if self.args.wloss_kldnpmul else torch.tensor(0.0)
            losses['kldnt'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['t_z'], n_z), dim=[1])) if self.args.wloss_kldntmul else torch.tensor(0.0)
                
            # -- total loss
            loss = losses['v2v'] * self.args.wloss_v2v + \
                losses['jts'] * self.args.wloss_jts + \
                losses['rot'] * self.args.wloss_rot + \
                losses['kldpt'] * self.args.wloss_kld + \
                losses['kldnp'] * self.args.wloss_kldnpmul * self.args.wloss_kld + \
                losses['kldnt'] * self.args.wloss_kldntmul * self.args.wloss_kld
            loss_value = loss.item()

            global_loss += loss_value
            global_step +=1

            
            # elbos (normalization is a bit different than for the losses)
            elbos = {}
            elbos[f'v2v'] = (-laplacian_nll(bm_orig.v, bm_rec.v, self.model.decsigma_v2v).sum()/2./bs - losses['kldpt']).detach().item()
            elbos[f'jts'] = (-laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, self.model.decsigma_jts).sum()/2./bs - losses['kldpt']).detach().item()
            elbos[f'rot'] = (-gaussian_nll(output[f'pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), self.model.decsigma_rot).sum()/2./bs - losses['kldpt']).detach().item()
            
        
            fid.add_gen_features( output['pose_body_text'] )



            ##LOGGER##
        logger.info("***** Running Validation *****")
        logger.info("loss: %d",  loss_value)
        for key in (elbos.keys()):
            logger.info("  %s = %s", key, str(elbos[key]))  
        for key in (losses.keys()):
            logger.info("  %s = %s", key, str(losses[key]))  

        fid_value = fid.compute()
        fidstr = fid.sstr()

        results = {fidstr: fid_value}
        results.update(losses)
        results.update(elbos)
            
        return results


    def evaluate(self):
    
        device = torch.device('cuda:0')
        fid_version = (self.args.fid, get_seed_from_model_path(self.args.model_path))
        suffix = get_epoch_from_model_path(self.args.model_path)

        torch.manual_seed(42)
        np.random.seed(42)

        if "posescript-A" in self.args.dataset:
            # average over captions
            results = {}
            nb_caps = len(config.caption_files[self.args.dataset])
            get_res_file = lambda cap_ind: os.path.join(os.path.dirname(self.args.model_path), f"result_{self.args.split}_{self.args.dataset}_X{fid_version[0]}-{fid_version[1]}X_{cap_ind}_{suffix}.txt")
            # load model if results for at least one caption is missing
            if sum([not os.path.isfile(get_res_file(cap_ind)) for cap_ind in range(nb_caps)]):
                # model, text_encoder_name = load_model(self.args.model_path, device)
                
                model, text_encoder_name = load_model(self.arg.model_path, device)

            # compute or load results for the given run & caption
            for cap_ind in range(nb_caps):
                filename_res = get_res_file(cap_ind)
                if os.path.isfilename(filename_res):
                    cap_results = load_from_file(filename_res)
                else:
                    d = PoseScript(version=self.args.dataset, split=self.args.split, text_encoder_name=text_encoder_name, caption_index=cap_ind, cache=True)
                    cap_results = compute_eval_metrics(model, d, fid_version, device)
                    save_to_file(cap_results, filename_res)

                # aggregate results
                results = {k:[v] for k, v in cap_results.items()} if not results else {k:results[k]+[v] for k,v in cap_results.items()}
            results = {k:sum(v)/nb_caps for k,v in results.items()}
        
        elif "posescript-H" in self.args.dataset:
            filename_res = os.path.join(os.path.dirname(self.args.model_path), f"result_{self.args.split}_{self.args.dataset}_X{fid_version[0]}-{fid_version[1]}X_{suffix}.txt")
            # compute or load results
            if os.path.isfile(filename_res):
                results = load_from_file

            else:
                model, text_encoder_name = load_model(self.arg.model_path, device)

                d = PoseScript(version=self.args.dataset, split=self.args.split, text_encoder_name=text_encoder_name, caption_index=0, cache=True)
                results = compute_eval_metrics(model, d, fid_version, device)
                save_to_file(results, filename_res)
            
        logger.info("***** Evaluation *****")
        for key in (results.keys()):
            logger.info("  %s = %s", key, str(results[key]))  
