import torch
from evaluation import calculate_murmur_scores
import numpy as np
import utils
classes = ['Present', 'Absent']
from torcheval.metrics.functional import binary_precision,binary_recall,binary_auprc, binary_auroc,binary_accuracy,binary_f1_score,binary_confusion_matrix


def evaluate(model, device, test_loader, loss_fn):
	correct = 0
	total = 0
	model.eval()
	target_all = []
	pred_all = []
	prob_all = []
	loss_avg = utils.RunningAverage()
	with torch.no_grad():
		for batch_idx, data in enumerate(test_loader):
			inputs = data[0].to(device)
			# target = data[1].squeeze(1).to(device)
			target = data[1].to(device)

			outputs = model(inputs)

			loss = loss_fn(outputs, target)
			loss_avg.update(loss.item())
			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()
			target_all.extend(target.cpu().numpy())
			pred_all.extend(predicted.cpu().numpy())
			prob_temp = torch.softmax(outputs, dim=1)
			prob_all.extend(prob_temp.data.cpu().numpy())
		y_test = np.zeros((total, 3))
		y_pred = np.zeros((total, 3))
		for i in range(total):
			y_test[i, target_all[i]] = 1
			y_pred[i, pred_all[i]] = 1
		score = calculate_murmur_scores(y_test, prob_all, y_pred)
	return loss_avg(), 100*correct/total, score, y_test, y_pred, prob_all


def evaluate_patch(model, device, test_loader, loss_fn):
	correct = 0
	total = 0
	model.eval()
	target_all = []
	pred_all = []
	prob_all = []
	loss_avg = utils.RunningAverage()
	with torch.no_grad():
		for batch_idx, data in enumerate(test_loader):
			inputs = data[0]
			inputs1 = data[1]
			# target = data[1].squeeze(1).to(device)
			# target = data[1].to(device)
			target = data[2].to(device)
			prob = []
			for idx, input in enumerate(inputs):
				outputs = model(input.to(device), inputs1[idx].to(device))
				loss = loss_fn(outputs, target)
				loss_avg.update(loss.item())
				# _, predicted = torch.max(outputs.data, 1)
				prob_temp = torch.softmax(outputs, dim=1)  # first use the average possibility of patches to classify
				prob.extend(prob_temp.data.cpu().numpy())
			prob_ave = np.mean(np.asarray(prob), axis=0)
			predicted = np.argmax(prob_ave)
			total += target.size(0)
			correct += (predicted == target).sum().item()
			target_all.extend(target.cpu().numpy())
			pred_all.append(predicted)#标签
			prob_all.append(prob_ave)#概率
		# y_test = np.zeros((total, 2))
		# y_pred = np.zeros((total, 2))
		# for i in range(total):
		# 	y_test[i, target_all[i]] = 1
		# 	y_pred[i, pred_all[i]] = 1
		# score = calculate_murmur_scores(y_test, np.asarray(prob_all), y_pred)
	labels_patients,target_patients=torch.tensor(pred_all),torch.tensor(target_all),
	prc_seg=binary_auprc(labels_patients,target_patients)
	roc_seg=binary_auroc(labels_patients,target_patients)
	acc_seg=binary_accuracy(labels_patients,target_patients)
	f1_seg=binary_f1_score(labels_patients,target_patients)
	ppv_seg=binary_precision(labels_patients,target_patients)
	trv_seg=binary_recall(labels_patients,target_patients)
	cm_seg=binary_confusion_matrix(labels_patients,target_patients)

	print(f'----segments_wise---- \n acc={acc_seg:.3%}\n roc:{roc_seg:.3f}\n prc:{prc_seg:.3f}\n f1:{f1_seg:.3f}\n ppv:{ppv_seg:.3f}\n recall:{trv_seg:.3f}\n cm:')
	print(cm_seg)
	return acc_seg,prc_seg,roc_seg,f1_seg,cm_seg#loss_avg(), 100 * correct / total, score, y_test, y_pred, prob_all
