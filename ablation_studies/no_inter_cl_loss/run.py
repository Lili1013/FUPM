import os
import sys
curPath = os.path.abspath(os.path.dirname((__file__)))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from utils.load_data import Load_Data
from utils.para_parser import parse
from models.local_model import Local_Model
from utils.params import data_params,model_params
from models.kmeans import k_means

from utils.evaluation import calculate_hr_ndcg

import os
import torch
from loguru import logger
from datetime import datetime
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
import numpy as np

def add_laplace_noise(data,noise_control):
    # self.beta = self.sensitivity/self.epsilon
    noise = torch.tensor(np.random.laplace(0, noise_control, data.shape))
    noise = noise.to(torch.float32)
    # noisy_data = data+noise
    return noise

def local_model_update(args, overlap_user_protos, model, local_data,optimizer,device):
    # logger.info('start train')
    model.train()

    total_client_loss, total_client_ce_loss, total_client_intra_cl_loss,total_client_intra_cl_loss_item, total_client_c_loss, total_client_disc_loss = [],[], [], [], [], []
    for iter in range(args.epochs):
        # logger.info('epoch : {}',iter)
        total_loss,total_ce_loss,total_intra_cl_loss,total_intra_cl_loss_item,total_c_loss, total_disc_loss= [],[],[],[],[],[]
        for i, data in enumerate(local_data.train_loader, 0):
            # logger.info(i)
            batch_nodes_u,batch_nodes_v,batch_nodes_c,batch_nodes_r = data
            batch_nodes_u, batch_nodes_v, categories_list, labels_list = local_data.generate_train_instances(batch_nodes_u.tolist(),
                                                                                            batch_nodes_v.tolist(),batch_nodes_c.tolist())
            batch_nodes_u, batch_nodes_v, categories_list,labels_list = torch.tensor(batch_nodes_u), torch.tensor(batch_nodes_v), \
                                                                        torch.tensor(categories_list,dtype=int),torch.tensor(labels_list)
            optimizer.zero_grad()
            pred_prob,u_id_feats,v_id_feats,u_review_feats,v_review_feats = model.forward(batch_nodes_u.to(device),batch_nodes_v.to(device),overlap_user_protos,local_data.user_inter_num_dict)
            L_ce = model.cross_entropy_loss(pred_prob,labels_list.to(device))
            # L_disc = model.disc_loss(u_common_feats,u_specific_feats)
            L_intral_cl_loss = model.calc_ssl_loss_intra(u_id_feats,v_id_feats,u_review_feats,v_review_feats)
            # L_intra_cl_loss_item = model.calc_ssl_loss_intra_item(v_id_feats,potential_item_id_feats)
            # if len(P)==0:
            #     L_C = 0*L_ce
            # else:
            #     L_C = model.calc_ssl_loss_proto(u_id_feats,u_feats)
                # L_C = model.proto_loss(P,overlap_user_protos,client_number,local_data.train_df,batch_nodes_u,categories_list.to(device),u_id_feats,cat_labels,args.client_num)
                # L_P, L_C = model.calculate_proto_loss(P, C, batch_nodes_u.tolist(), categories_list.tolist(), u_feats.to(torch.device('cpu')).detach().numpy())
            L = L_ce+args.gamma*L_intral_cl_loss
            # L = L_ce + args.alpha * L_C
            # logger.info('loss cross entropy: {}, loss P: {}, loss C:{}',L_ce.item(),L_P.item(),L_C.item())
            total_loss.append(L.item())
            total_ce_loss.append(L_ce.item())
            total_intra_cl_loss.append(L_intral_cl_loss.item())
            # total_c_loss.append(L_C.item())
            # total_intra_cl_loss_item.append(L_intra_cl_loss_item.item())
            # total_p_loss.append(L_P.item())
            # total_disc_loss.append(L_disc.item()
            L.backward(retain_graph=True)
            optimizer.step()
            # logger.info('loss:{}',L.item())
        total_client_loss.append(sum(total_loss) / len(total_loss))
        # total_client_disc_loss.append(sum(total_disc_loss)/len(total_disc_loss))
        total_client_ce_loss.append(sum(total_ce_loss) / len(total_ce_loss))
        total_client_intra_cl_loss.append(sum(total_intra_cl_loss)/len(total_intra_cl_loss))
        # total_client_intra_cl_loss_item.append(sum(total_intra_cl_loss_item)/len(total_intra_cl_loss_item))
        # total_client_p_loss.append(sum(total_p_loss) / len(total_p_loss))
        # total_client_c_loss.append(sum(total_c_loss) / len(total_c_loss))

    # #update prototypes
    # user_centroid_dict = update_local_protos(local_data=local_data,model=model,device=device,flag='')
    # local_data.update_cat(category_dict=cat_dict)
    return total_client_loss, total_client_ce_loss,total_client_intra_cl_loss
    # return total_client_loss, total_client_ce_loss, total_client_disc_loss,total_client_intra_cl_loss

def local_model_test(args,local_data,model,device,overlap_user_protos):
    model.eval()
    with torch.no_grad():
        # pos_samples,neg_samples = local_data.generate_test_instances(list(local_data.test_df['userID'].values),list(local_data.test_df['itemID'].values))
        pos_samples, neg_samples = local_data.test_pos_samples,local_data.test_neg_samples
        (hr, ndcg,mrr,hr_5,ndcg_5,mrr_5) = calculate_hr_ndcg(model=model,test_ratings=pos_samples,test_negatives=neg_samples,
                                           k=args.top_k,device=device,overlap_user_protos=overlap_user_protos,user_inter_nums=local_data.user_inter_num_dict)
        # hr = sum(hits) / len(hits)
        # ndcg = sum(ndcgs) / len(ndcgs)
    return hr,ndcg, mrr,hr_5,ndcg_5,mrr_5

if __name__ == '__main__':

    args = parse()
    logger.info(f'parameter settings: batch_size:{args.batch_size},lr:{args.lr},embed_id_dim:{args.embed_id_dim},'
                f'alpha:{args.alpha},gamma:{args.gamma}, beta:{args.beta},client_num:{args.client_num},'
                f'client_names:{args.client_names},'
                f'top_k:{args.top_k},local epochs:{args.epochs}')
    dataset = args.client_names[0].split('_')[0]
    date = f'{datetime.now().month}_{datetime.now().day}'
    train_model_weights_path,save_model_path = [],[]
    if args.client_num == 3:
        client_name_0 = args.client_names[0]
        client_name_1 = args.client_names[1]
        client_name_2 = args.client_names[2]
        save_model_path = [
f'best_models/{dataset}/{client_name_0}_emb_{args.embed_id_dim}_alpha_{args.alpha}_beta_{args.beta}_lr{args.lr}_b_{args.batch_size}_{date}.model',
f'best_models/{dataset}/{client_name_1}_emb_{args.embed_id_dim}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}_b_{args.batch_size}_{date}.model',
f'best_models/{dataset}/{client_name_2}_emb_{args.embed_id_dim}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}_b_{args.batch_size}_{date}.model']

        train_model_weights_path = [
f'best_models/{dataset}/train_model_weights_epochs_{args.client_names[0]}_emb_{args.embed_id_dim}__alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}_b_{args.batch_size}_epoch_{args.epochs}/',
f'best_models/{dataset}/train_model_weights_epochs_{args.client_names[1]}_emb_{args.embed_id_dim}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}_b_{args.batch_size}_epoch_{args.epochs}/',
f'best_models/{dataset}/train_model_weights_epochs_{args.client_names[2]}_emb_{args.embed_id_dim}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}_b_{args.batch_size}_epoch_{args.epochs}/']
    if args.client_num == 2:
        client_name_0 = args.client_names[0]
        client_name_1 = args.client_names[1]
        save_model_path = [
    f'best_models/{dataset}/{client_name_0}_emb_{args.embed_id_dim}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}_b_{args.batch_size}_{date}.model',
    f'best_models/{dataset}/{client_name_1}_emb_{args.embed_id_dim}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}_b_{args.batch_size}_{date}.model']
        train_model_weights_path = [
    f'best_models/{dataset}/train_model_weights_epochs_{args.client_names[0]}_emb_{args.embed_id_dim}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}_b_{args.batch_size}_epoch_{args.epochs}/',
    f'best_models/{dataset}/train_model_weights_epochs_{args.client_names[1]}_emb_{args.embed_id_dim}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}_b_{args.batch_size}_epoch_{args.epochs}/']
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(1)

    local_model_list = []
    local_data_list = []
    local_optimizer_list = []
    pca = PCA(n_components=args.embed_id_dim)
    # save_model_paths = args.save_model_path
    #initializa local models and local data objects
    logger.info('initialize local data and local model:{}',args.client_names)
    for j in range(args.client_num):
        data_params[args.client_names[j]].update({'args': args})
        local_data = Load_Data(**data_params[args.client_names[j]])
        local_data_list.append(local_data)
        field = args.client_names[j].split('_')[0]
        if field == 'douban':
            user_review_emb = torch.tensor(pca.fit_transform(local_data.user_review_emb))
            item_review_emb = torch.tensor(pca.fit_transform(local_data.item_review_emb))
        else:
            user_review_emb = local_data.user_review_emb
            item_review_emb = local_data.item_review_emb
        # client_name = args.client_names[j].split('_')[0]+'_'+ args.client_names[j].split('_')[-1]
        model_params[args.client_names[j]].update({
            'tau': args.tau,
            'beta':args.beta,
            'embed_id_dim': args.embed_id_dim,
            'device': device,
            'train_data': local_data.train_df,
            'review_embed_dim': args.review_embed_dim,
            'u_review_feat': user_review_emb,
            'v_review_feat': item_review_emb,
            'n_layers': args.n_layers,
            'potential_pos_dict':local_data.potential_pos_dict,
            'local_data':local_data,
            # 'u_neg_dict':local_data.u_neg_dict
        })
        # logger.info(client_name)
        local_model = Local_Model(**model_params[args.client_names[j]])
        local_model_list.append(local_model.to(device))
        local_optimizer_list.append(torch.optim.Adam(local_model.parameters(), lr=args.lr,weight_decay=args.l2_regularization))

    P,P_new = {},{}  # local prototypes for each client
    C,C_new = {},{}
    overlap_user_protos = {}# global prototypes
    labels = set()

    best_hr_1, best_ndcg_1, best_mrr_1, best_hr_2, best_ndcg_2, best_mrr_2,best_hr_3, best_ndcg_3, best_mrr_3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0
    endure_count_1 = 0
    endure_count_2 = 0
    endure_count_3 = 0
    endure_count = 0
    best_hr_avg,best_ndcg_avg,best_mrr_avg = 0.0,0.0,0.0

    for i in range(args.rounds):
        if endure_count > 5:
            logger.info('End')
            logger.info('best hr: {}, best ndcg : {},best mrr : {}',best_hr_avg,best_ndcg_avg,best_mrr_avg)
            break

        hrs,ndcgs,mrrs=[],[],[]
        for j in range(args.client_num):
            # logger.info('round: {}, client {}',i,j)
            total_client_loss, total_client_ce_loss,total_client_intra_cl_loss = local_model_update(args=args,
                                            overlap_user_protos=overlap_user_protos,model=local_model_list[j], local_data=local_data_list[j],
                                            optimizer=local_optimizer_list[j],device
                                          =device)
            hr,ndcg,mrr,hr_5,ndcg_5,mrr_5=local_model_test(args=args,local_data=local_data_list[j],model=local_model_list[j],device=device,overlap_user_protos=overlap_user_protos)
            # if not os.path.exists(train_model_weights_path[j]):
            #     os.makedirs(train_model_weights_path[j])
            # torch.save(local_model_list[j].state_dict(), '{}/weights_epoch_{}.pth'.format(train_model_weights_path[j], i))
            logger.info(
                'round:%d, client:%d, loss:%.5f, ce loss:%.5f, intra cl loss:%.5f,'
                'hr_5:%.5f, ndcg_5:%.5f,mrr_5:%.5f,hr_10:%.5f, ndcg_10:%.5f,mrr_10:%.5f' % (
                    i, j,
                    sum(total_client_loss) / len(total_client_loss),
                    sum(total_client_ce_loss) / len(total_client_ce_loss),
                    # sum(total_client_disc_loss) / len(total_client_disc_loss),
                    sum(total_client_intra_cl_loss)/len(total_client_intra_cl_loss),
                    # sum(total_client_intra_cl_loss_item)/len(total_client_intra_cl_loss_item),
                    # sum(total_client_c_loss) / len(total_client_c_loss),
                    hr_5,ndcg_5,mrr_5,
                     hr, ndcg,mrr))

            hrs.append(hr)
            ndcgs.append(ndcg)
            mrrs.append(mrr)
        hr_avg = sum(hrs) / len(hrs)
        ndcg_avg = sum(ndcgs) / len(ndcgs)
        mrr_avg = sum(mrrs)/len(mrrs)
        if hr_avg >= best_hr_avg:
            best_hr_avg = hr_avg
            best_ndcg_avg = ndcg_avg
            best_mrr_avg = mrr_avg
            endure_count = 0
            # if args.client_num == 3:
            #     torch.save(local_model_list[0].state_dict(), '{}'.format(save_model_path[0]))
            #     torch.save(local_model_list[1].state_dict(), '{}'.format(save_model_path[1]))
            #     torch.save(local_model_list[2].state_dict(), '{}'.format(save_model_path[2]))
            # else:
            #     torch.save(local_model_list[0].state_dict(), '{}'.format(save_model_path[0]))
            #     torch.save(local_model_list[1].state_dict(), '{}'.format(save_model_path[1]))
        else:
            endure_count += 1

        # if hrs[0] >= best_hr_1:
        #     best_hr_1 = hrs[0]
        #     best_ndcg_1 = ndcgs[0]
        #     best_mrr_1 = mrrs[0]
        #     endure_count_1 = 0
        # else:
        #     endure_count_1 += 1
        # if hrs[1] >= best_hr_2:
        #     best_hr_2 = hrs[1]
        #     best_ndcg_2 = ndcgs[1]
        #     best_mrr_2 = mrrs[1]
        #     endure_count_2 = 0
        # else:
        #     endure_count_2 +=1
        # if args.client_num == 3:
        #     if hrs[2] >= best_hr_3:
        #         best_hr_3 = hrs[2]
        #         best_ndcg_3 = ndcgs[2]
        #         best_mrr_3 = mrrs[2]
        #         endure_count_3 = 0
        #     else:
        #         endure_count_3 +=1

        # logger.info('server agg')
        # overlap_user_protos = server_update(P=P,local_data_list=local_data_list,
        #                             overlap_user_num=local_data_list[0].overlap_user_num,
        #                             global_proto_agg_way=args.global_proto_agg_way)

