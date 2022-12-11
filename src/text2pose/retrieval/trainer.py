import os
import shutil
import time
import numpy as np
from tqdm import tqdm, trange
import logging
import torch 
import json
# import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

import text2pose.config as config
from text2pose.vocab import Vocabulary # needed
from text2pose.data import PoseScript
from text2pose.retrieval.our_model_retrieval import PoseText
from text2pose.utils import save_to_file, load_from_file


logger = logging.getLogger(__name__)

def load_model(model_path, device):
	
	ckpt = torch.load(model_path, 'cpu')
	text_encoder_name = ckpt['args'].text_encoder_name
	latentD = ckpt['args'].latentD

	model = PoseText(text_encoder_name=text_encoder_name, latentD=latentD).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print("Loaded model from:", model_path)

	return model, text_encoder_name


def compute_eval_metrics(model, dataset, device, compute_loss=False):

    poses_features, texts_features = infer_features(model, dataset, device)
	
    p2t_recalls = x2y_metrics(poses_features, texts_features, config.k_recall_values, sstr="p2t_")

    t2p_recalls = x2y_metrics(texts_features, poses_features, config.k_recall_values, sstr="t2p_")

    recalls = {"mRecall": (sum(p2t_recalls.values()) + sum(t2p_recalls.values())) / (2 * len(config.k_recall_values))}
    recalls.update(p2t_recalls)
    recalls.update(t2p_recalls)

    if compute_loss:
        score_t2p = texts_features.mm(poses_features.t())
        # BBC loss
        scores = score_t2p*model.loss_weight
        batch_size = scores.shape[0]
        GT_labels = torch.arange(batch_size).long()
        GT_labels = torch.autograd.Variable(GT_labels)
        if torch.cuda.is_available():
            GT_labels = GT_labels.cuda()

        loss = F.cross_entropy(scores, GT_labels)

        loss_value = loss.item()
        return recalls, loss_value

    return recalls


def infer_features(model, dataset, device):

	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=32,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)

	poses_features = torch.zeros(len(dataset), model.latentD).to(device)
	texts_features = torch.zeros(len(dataset), model.latentD).to(device)

	for i, batch in tqdm(enumerate(data_loader)):
		poses = batch['pose'].to(device)
		caption_tokens = batch['caption_tokens'].to(device)
		caption_lengths = batch['caption_lengths'].to(device)
		caption_tokens = caption_tokens[:,:caption_lengths.max()]
		with torch.inference_mode():
			pfeat, tfeat = model(poses, caption_tokens, caption_lengths)
			poses_features[i*32:i*32+len(poses)] = pfeat
			texts_features[i*32:i*32+len(poses)] = tfeat

	return poses_features, texts_features


def x2y_metrics(x_features, y_features, k_values, sstr=""):

	# initialize metrics
	nb_x = len(x_features)
	sstrR = sstr + 'R@%d'
	recalls = {sstrR%k:0 for k in k_values}

	# evaluate for each query x
	for x_ind in tqdm(range(nb_x)):

		scores = x_features[x_ind].view(1, -1).mm(y_features.t())[0].cpu()

		_, indices_rank = scores.sort(descending=True)

		GT_rank = torch.where(indices_rank == x_ind)[0][0].item()
		for k in k_values:
			recalls[sstrR%k] += GT_rank < k

	# average metrics
	recalls = {sstrR%k: recalls[sstrR%k]/nb_x*100.0 for k in k_values}
	return recalls

class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = "cuda:0"
        # self.seed = args.seed
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        if args.mode == "train":
            self.model = PoseText(text_encoder_name=args.text_encoder_name, latentD=args.latentD)
            self.model.to(self.device)


        self.generated_pose_samples_path = None # default
        if args.generated_pose_samples:
            generated_pose_samples_model_path = (config.shortname_2_model_path[args.generated_pose_samples]).format(seed=args.seed)
            self.generated_pose_samples_path = config.generated_pose_path % os.path.dirname(generated_pose_samples_model_path)





    def train(self):
        ckpt_fname = os.path.join(self.args.output_dir, 'checkpoint_last.pth')

        print('Load training dataset')
        dataset_train = PoseScript(version=self.args.dataset, split='train', text_encoder_name=self.args.text_encoder_name,
                                     caption_index='rand', generated_pose_samples_path=self.generated_pose_samples_path)
                                     
        # print(dataset_train, len(dataset_train))
        train_dataloader = DataLoader(
            dataset_train, sampler=None, shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print('Load validation dataset')
        val_dataloader = PoseScript(version=self.args.dataset, split='val', text_encoder_name=self.args.text_encoder_name, 
                                    caption_index="deterministic-mix", generated_pose_samples_path=self.generated_pose_samples_path)

        # print(dataset_val, len(dataset_val))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

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
            self.model.train()
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                
                
                # batch = tuple(t.to(self.args.device) for t in batch)
                poses = batch['pose'].to(self.device)
                caption_tokens = batch['caption_tokens'].to(self.device)
                caption_lengths = batch['caption_lengths'].to(self.device)
                caption_tokens = caption_tokens[:,:caption_lengths.max()]# truncate within the batch, based on the longest text 
                
                # compute scores
                poses_features, texts_features = self.model(poses, caption_tokens, caption_lengths)
                score_t2p = texts_features.mm(poses_features.t())



                #COMPUTE LOSS
                scores = score_t2p*self.model.loss_weight
                batch_size = scores.shape[0]
                GT_labels = torch.arange(batch_size).long()
                GT_labels = torch.autograd.Variable(GT_labels)
                if torch.cuda.is_available():
                    GT_labels = GT_labels.cuda()
                
                loss = F.cross_entropy(scores, GT_labels) # mean reduction


                loss_value = loss.item()
                global_loss += loss_value
                global_step +=1
                
                # step
                # scheduler.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ##LOGGER##
            logger.info("***** Running training *****")
            logger.info("Epoch: %d, loss: %d", epoch, loss_value)

            val_results = self.validate(val_dataloader)

                # save checkpoint
            tosave = {'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                    'args': self.args,
                    'best_score': best_score}
            # ... current checkpoint
            torch.save(tosave, ckpt_fname)
            
            if (not best_score) or (val_results["mRecall"] > best_score):
                best_score = val_results["mRecall"]
                shutil.copyfile(ckpt_fname, os.path.join(self.args.output_dir, 'best_model.pth'))

            log_stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]["lr"]}
            log_stats.update(val_results)
            with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        logger.info("**********Training Finished***********")
        logger.info("Training Time: %d", time.time() - start_time)
        # return global_loss / global_step


    def validate(self, dataloader):    
        self.model.eval()  
        recalls, loss_value = compute_eval_metrics(self.model, dataloader, self.device, compute_loss=True)
        results = {"val_loss": loss_value}
        results.update(recalls)

        logger.info("***** Validation *****")
        for key in (results.keys()):
            logger.info("  %s = %s", key, str(results[key]))    
            
        return results

    
    def evaluate(self):

        device = torch.device('cuda:0')
        precision = ""
        generated_pose_samples_path = None
        if self.args.generated_pose_samples:
            precision = f"gensample_{self.args.generated_pose_samples}_"
            seed = self.args.model_path.split("seed")[1].split("/")[0]
            generated_pose_samples_model_path = (config.shortname_2_model_path[self.args.generated_pose_samples]).format(seed=seed)
            generated_pose_samples_path = config.generated_pose_path % os.path.dirname(generated_pose_samples_model_path)
        
        if "posescript-A" in self.args.dataset:
            results = {}
            nb_caps = len(config.caption_files[self.args.dataset])
            get_res_file = lambda cap_ind: os.path.join(os.path.dirname(self.args.model_path), f"result_{self.args.split}_{precision}{self.args.dataset}_{cap_ind}.txt")

            if sum([not os.path.isfile(get_res_file(cap_ind)) for cap_ind in range(nb_caps)]):
                model, text_encoder_name = load_model(self.args.model_path, device)

            for cap_ind in range(nb_caps):
                filename_res = get_res_file(cap_ind)
                if os.path.isfile(filename_res):
                    cap_results = load_from_file(filename_res)
                else:
                    d = PoseScript(version=self.args.dataset, split=self.args.split, text_encoder_name=text_encoder_name, caption_index=cap_ind, cache=True, generated_pose_samples_path=generated_pose_samples_path)
                    cap_results = compute_eval_metrics(model, d, device)
                    save_to_file(cap_results, filename_res)
                # aggregate results
                results = {k:[v] for k, v in cap_results.items()} if not results else {k:results[k]+[v] for k,v in cap_results.items()}
            results = {k:sum(v)/nb_caps for k,v in results.items()}
        
        else:
            filename_res = os.path.join(os.path.dirname(self.args.model_path), f"result_{self.args.split}_{precision}{self.args.dataset}.txt")
            if os.path.isfile(filename_res):
                results = load_from_file(filename_res)
            else:
                model, text_encoder_name = load_model(self.args.model_path, device)

                d = PoseScript(version=self.args.dataset, split=self.args.split, text_encoder_name=text_encoder_name, caption_index=0, cache=True, generated_pose_samples_path=generated_pose_samples_path)
                results = compute_eval_metrics(model, d, device)
                save_to_file(results, filename_res)
            
        logger.info("***** Evaluation *****")
        for key in (results.keys()):
            logger.info("  %s = %s", key, str(results[key]))  
