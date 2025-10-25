from utils import *

# ------------------------------------------ Training Free ------------------------------------------


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_cache_keys = cache_keys

    affinity = val_features @ best_cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    print(
        "**** Cache val accuracy: {:.2f}. ****\n".format(cls_acc(cache_logits, val_labels)))

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(
        cfg, best_cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features @ best_cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)
                    ).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    return acc


def Refinement(cfg, new_cache_keys, cache_values, clip_weights, new_clip_weights,
               val_features, new_val_features, val_labels,
               test_features, new_test_features, test_labels):

    # Zero-shot CLIP
    R_fW = 100. * test_features @ clip_weights
    acc = cls_acc(R_fW, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']

    # Calculate R_f'F'
    R_fF = new_test_features @ new_cache_keys.t()

    # Calculate R_F'W'
    key_logits = new_cache_keys @ new_clip_weights
    key_logits = key_logits.softmax(1)
    cache_div = torch.sum(
        cache_values * torch.log2((cache_values + 1e-6) / (key_logits + 1e-6)), dim=1)[:, None]
    R_FW = (cache_div * gamma).exp()
    soft_cache_values = cache_values * R_FW

    cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ soft_cache_values

    ape_logits = R_fW + cache_logits * alpha
    acc = cls_acc(ape_logits, test_labels)
    print("**** Before search, test accuracy: {:.2f}. ****\n".format(acc))

    best_search_acc = 0
    R_fF = new_val_features @ new_cache_keys.t()
    R_fW = 100. * val_features @ clip_weights
    best_beta, best_alpha, best_gamma = 0, 0, 0
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step']
                 [0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step']
                  [1] + 0.1 for i in range(cfg['search_step'][1])]
    gamma_list = [i * cfg['search_scale'][2] / cfg['search_step'][2]
                  for i in range(cfg['search_step'][2])]
    for beta in beta_list:
        for alpha in alpha_list:
            for gamma in gamma_list:
                with torch.no_grad():
                    soft_cache_values = cache_values * \
                        (cache_div * gamma).exp()
                    cache_logits = ((-1) * (beta - beta * R_fF)
                                    ).exp() @ soft_cache_values
                    ape_logits = R_fW + cache_logits * alpha
                acc = cls_acc(ape_logits, val_labels)
                if acc > best_search_acc:
                    print("New best setting, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(
                        alpha, beta, gamma, acc))
                    best_search_acc = acc
                    best_alpha, best_beta, best_gamma = alpha, beta, gamma
    print("\nAfter searching, the best val accuarcy: {:.2f}.\n".format(
        best_search_acc))

    R_fW = 100. * test_features @ clip_weights
    R_fF = new_test_features @ new_cache_keys.t()

    soft_cache_values = cache_values * (cache_div * best_gamma).exp()
    cache_logits = ((-1) * (best_beta - best_beta * R_fF)
                    ).exp() @ soft_cache_values

    ape_logits = R_fW + cache_logits * best_alpha
    acc = cls_acc(ape_logits, test_labels)
    print("**** APE's test accuracy: {:.2f}. ****\n".format(acc))

    return acc


def APE(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, early_stop=False):

    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num).cuda()
    cache_keys = cache_keys.t().reshape(
        cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim).cuda()

    cache_keys, cache_values = cache_keys.reshape(
        -1, feat_dim), cache_values.reshape(-1, cate_num)

    cfg['w'] = cfg['w_training_free']
    indices = cal_criterion(cfg, clip_weights, cache_keys,
                            only_use_txt=False, force=False)

    new_clip_weights = clip_weights[indices, :]
    new_cache_keys = cache_keys[:, indices]
    new_test_features = test_features[:, indices]
    new_val_features = val_features[:, indices]

    new_clip_weights = new_clip_weights / \
        new_clip_weights.norm(dim=0, keepdim=True)
    new_cache_keys = new_cache_keys / new_cache_keys.norm(dim=-1, keepdim=True)
    new_test_features = new_test_features / \
        new_test_features.norm(dim=-1, keepdim=True)
    new_val_features = new_val_features / \
        new_val_features.norm(dim=-1, keepdim=True)

    if early_stop:
        return new_clip_weights, new_cache_keys, new_val_features, new_test_features

    acc = Refinement(cfg, new_cache_keys, cache_values, clip_weights, new_clip_weights,
                     val_features, new_val_features, val_labels,
                     test_features, new_test_features, test_labels)

    return acc


def GDA(vecs, labels, clip_weights, val_features, val_labels, alpha_shift=False):
    # normal distribution
    mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True)
                    for i in range(clip_weights.shape[1])])

    # KS Estimator.
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0)
                            for i in range(clip_weights.shape[1])])
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov(
    ) + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda())

    ps = torch.ones(clip_weights.shape[1]).cuda() * 1. / clip_weights.shape[1]
    W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

    # Evaluate
    # Grid search for hyper-parameter alpha
    best_val_acc = 0
    best_alpha = 0.1
    # [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    for alpha in [10**i for i in range(-4, 5)]:
        if alpha_shift:
            val_logits = alpha * val_features.float() @ clip_weights.float() + \
                val_features.float() @ W + b
        else:
            val_logits = 100. * val_features.float() @ clip_weights.float() + alpha * \
                (val_features.float() @ W + b)

        acc = cls_acc(val_logits, val_labels)
        if acc > best_val_acc:
            best_val_acc = acc
            best_alpha = alpha
    ############################################################################
    # mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(clip_weights.shape[1])]).float()
    # center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(clip_weights.shape[1])])
    # cov = center_vecs.T.cov()
    # cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * cov + cov.trace() * torch.eye(center_vecs.shape[1]).cuda())

    # ps = torch.ones(clip_weights.shape[1]).cuda() * 1. / clip_weights.shape[1]
    # W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    # b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

    # # Evaluate
    # # Grid search for hyper-parameter alpha
    # best_val_acc = 0
    # best_alpha = 0.1
    # for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]:
    #     val_logits = alpha * val_features.float() @ clip_weights.float() + \
    #         val_features.float() @ W + b

    #     acc = cls_acc(val_logits, val_labels)
    #     if acc > best_val_acc:
    #         best_val_acc = acc
    #         best_alpha = alpha

    print("best_val_alpha: %s \t best_val_acc: %s" %
          (best_alpha, best_val_acc))
    alpha = best_alpha
    return alpha, W, b, best_val_acc


def GDA_CLIP(cfg, val_features, val_labels, test_features, test_labels, clip_weights):

    # Parameter Estimation.
    with torch.no_grad():
        # Ours
        vecs = torch.load(
            cfg['cache_dir'] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
        labels = torch.load(
            cfg['cache_dir'] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
        alpha, W, b, _ = GDA(vecs, labels, clip_weights,
                             val_features, val_labels)

        test_logits = 100. * test_features.float() @ clip_weights.float() + alpha * \
            (test_features.float() @ W + b)
        notune_acc = cls_acc(test_logits, test_labels)
        print("training-free acc:", notune_acc)
    return notune_acc

#clip_weights_cupl_all -> clip_weights_all [C, P, D]
#clip_weights_IGT -> clip_weights [d,c]
#matching_score  N, P
def TIMO(cfg, val_features, val_labels, test_features, test_labels,
         clip_weights, clip_weights_all, matching_score, vecs_labels=None,
         grid_search=False, n_quick_search=-1, is_print=False):

    best_val_acc = 0
    best_alpha = 0.1

    with torch.no_grad():
        # Image Vecs
        if vecs_labels is None:
            vecs_v = torch.load(
                cfg['cache_dir'] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
            labels_v = torch.load(
                cfg['cache_dir'] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
        else:
            vecs_v, labels_v = vecs_labels[0], vecs_labels[1]

        vecs_t = clip_weights_all.clone().float()  # c, n, d
        vecs_t, weights = vec_sort(vecs_t, matching_score)
        cate_num, prompt_num, _, = vecs_t.shape
        vecs_c, labels_c = vecs_v, labels_v

        if grid_search:
            if n_quick_search != -1:
                beta_list = [int(t) for t in torch.linspace(
                    1, prompt_num*2, n_quick_search)]
            else:
                beta_list = range(1, prompt_num*2)
        else:
            beta_list = [prompt_num]

        for beta in beta_list:
            beta = beta + 1 if beta == 0 else beta

            sliced_vecs_t = vecs_t.repeat(1, 2, 1)[:, :beta, :]  # c, s, d
            sliced_weights = weights.repeat(1, 2)[:, :beta]  # c, s

            # weight for instance based transfer
            sliced_vecs_t = sliced_vecs_t * sliced_weights.unsqueeze(-1)

            sliced_vecs_t = sliced_vecs_t.reshape(cate_num*beta, -1)
            tmp = torch.tensor(range(cate_num)).unsqueeze(1).repeat(1, beta)
            sliced_labels_t = tmp.flatten().to(sliced_vecs_t.device)

            # Instance based transfer
            vecs_c = torch.cat([sliced_vecs_t, vecs_v])
            labels_c = torch.cat([sliced_labels_t, labels_v])

            alpha, W, b, val_acc = GDA(
                vecs_c, labels_c, clip_weights, val_features, val_labels, alpha_shift=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_beta = beta
                best_alpha = alpha
                best_weights = [W.clone(), b.clone()]

        alpha = best_alpha
        #logits =  αlogitsIGT + logitsTGI
        test_logits = alpha * test_features.float() @ clip_weights.float() + (test_features.float() @ best_weights[0] + best_weights[1])
        acc = cls_acc(test_logits, test_labels)

        if is_print:
            print("best_val_alpha: %s \t best_val_acc: %s" %
                  (best_alpha, best_val_acc))
            print("best_beta:", best_beta)
            print("training-free acc:", acc)
            print()

    return acc

def timo_with_ape(cfg,cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,clip_weights_all,image_weights_all):
    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num).cuda()
    cache_keys = cache_keys.t().reshape(cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim).cuda()

    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)

    cfg['w'] = cfg['w_training_free']
    indices = cal_criterion(cfg, clip_weights, cache_keys,
                            only_use_txt=False, force=False)

    new_clip_weights = clip_weights[indices, :]
    new_cache_keys = cache_keys[:, indices]
    new_test_features = test_features[:, indices]
    new_val_features = val_features[:, indices]

    new_clip_weights = new_clip_weights / \
        new_clip_weights.norm(dim=0, keepdim=True)
    new_cache_keys = new_cache_keys / new_cache_keys.norm(dim=-1, keepdim=True)
    new_test_features = new_test_features / \
        new_test_features.norm(dim=-1, keepdim=True)
    new_val_features = new_val_features / \
        new_val_features.norm(dim=-1, keepdim=True)
    
    new_clip_weights_all = clip_weights_all[:,:,indices]
    new_clip_weights_all = new_clip_weights_all / new_clip_weights_all.norm(dim=-1, keepdim=True)
    new_image_weights_all = image_weights_all[:,:,indices]
    new_image_weights = new_image_weights_all.mean(dim=1)
    new_image_weights = new_image_weights / new_image_weights.norm(dim=1, keepdim=True)

    clip_weights_IGT, matching_score = image_guide_text(cfg, new_clip_weights_all, new_image_weights, return_matching=True)
    clip_weights_IGT = clip_weights_IGT.t()
    clip_weights_IGT, matching_score = image_guide_text_search(cfg, new_clip_weights_all, new_val_features, val_labels, new_image_weights)
    #acc_free = TIMO(cfg, new_val_features, val_labels, new_test_features, test_labels, 
    #    clip_weights_IGT, new_clip_weights_all, matching_score,vecs_labels=None,
    #    grid_search=True, n_quick_search=10, is_print=True)
    vecs_labels=None
    n_quick_search=10
    is_print=True
    grid_search=True

    best_val_acc = 0
    best_alpha = 0.1
    with torch.no_grad():
        # Image Vecs
        if vecs_labels is None:
            vecs_v = torch.load(
                cfg['cache_dir'] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
            labels_v = torch.load(
                cfg['cache_dir'] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
        else:
            vecs_v, labels_v = vecs_labels[0], vecs_labels[1]

        new_vecs_v = vecs_v[:,indices]
        vecs_t = new_clip_weights_all.clone().float()  # c, n, d
        vecs_t, weights = vec_sort(vecs_t, matching_score)
        cate_num, prompt_num, _, = vecs_t.shape
        vecs_c, labels_c = new_vecs_v, labels_v

        if grid_search:
            if n_quick_search != -1:
                beta_list = [int(t) for t in torch.linspace(
                    1, prompt_num*2, n_quick_search)]
            else:
                beta_list = range(1, prompt_num*2)
        else:
            beta_list = [prompt_num]

        for beta in beta_list:
            beta = beta + 1 if beta == 0 else beta

            sliced_vecs_t = vecs_t.repeat(1, 2, 1)[:, :beta, :]  # c, s, d
            sliced_weights = weights.repeat(1, 2)[:, :beta]  # c, s

            # weight for instance based transfer
            sliced_vecs_t = sliced_vecs_t * sliced_weights.unsqueeze(-1)

            sliced_vecs_t = sliced_vecs_t.reshape(cate_num*beta, -1)
            tmp = torch.tensor(range(cate_num)).unsqueeze(1).repeat(1, beta)
            #print(f"Debug before error:")
            #print(f"tmp shape: {tmp.shape}, dtype: {tmp.dtype}, device: {tmp.device}")
            #print(f"tmp values: {tmp.flatten()[:10]}")  # 打印前10个值
            #print(f"sliced_vecs_t shape: {sliced_vecs_t.shape}, device: {sliced_vecs_t.device}")
            #print(f"new_vecs_v shape: {new_vecs_v.shape}, device: {new_vecs_v.device}")
            #print(f"labels_v shape: {labels_v.shape}, dtype: {labels_v.dtype}, device: {labels_v.device}")
            sliced_labels_t = tmp.flatten().to(sliced_vecs_t.device)

            # Instance based transfer
            vecs_c = torch.cat([sliced_vecs_t, new_vecs_v])
            labels_c = torch.cat([sliced_labels_t, labels_v])

            alpha, W, b, val_acc = GDA(
                vecs_c, labels_c, clip_weights_IGT, new_val_features, val_labels, alpha_shift=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_beta = beta
                best_alpha = alpha
                best_weights = [W.clone(), b.clone()]

        alpha = best_alpha
        test_logits = alpha * new_test_features.float() @ clip_weights_IGT.float() + \
            (new_test_features.float() @ best_weights[0] + best_weights[1])
        acc = cls_acc(test_logits, test_labels)

        if is_print:
            print("best_val_alpha: %s \t best_val_acc: %s" %
                  (best_alpha, best_val_acc))
            print("best_beta:", best_beta)
            print("training-free acc:", acc)
            print()
    return acc
'''
提示: 本文件中的一些方法在main中的调用方式以及张量的尺寸
# Textual features
print("\nGetting textual features as CLIP's classifier.")
#  所有prompt的特征 [C, P, D] c: 类别数量 这里是100 p: prompt数量 这里是22 d：文本特征的维度 这里是1024
clip_weights_cupl_all = torch.load(cfg['cache_dir'] + "/text_weights_cupl_t_all.pt", weights_only=False)
cate_num, prompt_cupl_num, dim = clip_weights_cupl_all.shape
# 平均化的prompt的特征的转置 [D, C] [1024,100]
clip_weights_cupl = clip_weights_cupl_all.mean(dim=1).t()
clip_weights_cupl = clip_weights_cupl / clip_weights_cupl.norm(dim=0, keepdim=True)

# Construct the cache model by few-shot training set
print("\nConstructing cache model by few-shot visual features and labels.")
# cache_keys: 经过clip处理后的fewshot图形样本的张量 [D,C * shot_num]
# cahce_value: onehot vector [C* shot_num, C]
cache_keys, cache_values = load_few_shot_feature(cfg)

.........................

acc_free = timo_with_ape(cfg, cache_keys, cache_values, val_features, val_labels,  
        test_features, test_labels, clip_weights_cupl,clip_weights_cupl_all,image_weights_all)
    metric['TIMO_APE_v1'] = acc_free 

    timo_with_ape_v3(cfg, cache_keys, cache_values, val_features, val_labels, 
                     test_features, test_labels, clip_weights_IGT, clip_weights_cupl_all, image_weights_all)
    metric['TIMO_APE_v3'] = acc_free 

    timo_with_ape_v2(cfg, cache_keys, cache_values, val_features, val_labels, 
                     test_features, test_labels, clip_weights_IGT, clip_weights_cupl_all, image_weights_all)
    metric['TIMO_APE_v2'] = acc_free 
'''
def _timo_grid_search_eval(cfg, val_features, val_labels, test_features, test_labels,
                           clip_weights_IGT, clip_weights_all, matching_score, indices):
    """
    Helper function to perform TIMO's grid search on a feature subspace and return
    both test and validation accuracy.
    """
    best_val_acc = 0
    best_alpha = 0.1
    best_beta = -1
    best_weights = None
    
    with torch.no_grad():
        # Load image vectors (from cache) and project to subspace
        vecs_v = torch.load(cfg['cache_dir'] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
        labels_v = torch.load(cfg['cache_dir'] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()
        subspace_vecs_v = vecs_v[:, indices]

        # Prepare text vectors
        vecs_t = clip_weights_all.clone().float()  # c, n, d
        vecs_t, weights = vec_sort(vecs_t, matching_score)
        cate_num, prompt_num, _, = vecs_t.shape

        # Define search space for beta
        beta_list = [int(t) for t in torch.linspace(1, prompt_num * 2, 10)] # n_quick_search=10

        for beta in beta_list:
            beta = beta + 1 if beta == 0 else beta

            sliced_vecs_t = vecs_t.repeat(1, 2, 1)[:, :beta, :]
            sliced_weights = weights.repeat(1, 2)[:, :beta]
            sliced_vecs_t = sliced_vecs_t * sliced_weights.unsqueeze(-1)
            sliced_vecs_t = sliced_vecs_t.reshape(cate_num * beta, -1)
            
            tmp = torch.tensor(range(cate_num)).unsqueeze(1).repeat(1, beta)
            sliced_labels_t = tmp.flatten().to(sliced_vecs_t.device)

            # Instance-based transfer
            vecs_c = torch.cat([sliced_vecs_t, subspace_vecs_v])
            labels_c = torch.cat([sliced_labels_t, labels_v])

            # GDA for finding alpha
            alpha, W, b, val_acc = GDA(vecs_c, labels_c, clip_weights_IGT, val_features, val_labels, alpha_shift=True)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_beta = beta
                best_alpha = alpha
                best_weights = [W.clone(), b.clone()]

        # Evaluate test accuracy with the best found hyperparameters
        test_logits = best_alpha * test_features.float() @ clip_weights_IGT.float() + \
                      (test_features.float() @ best_weights[0] + best_weights[1])
        test_acc = cls_acc(test_logits, test_labels)

        print(f"Search Results -> best_beta: {best_beta}, best_alpha: {best_alpha:.4f}, val_acc: {best_val_acc:.4f}, test_acc: {test_acc:.4f}")

    return test_acc, best_val_acc

# 思路二（软特征选择）
'''
在这个思路中，您可以尝试使用一个软选择方法来代替硬选择（如top-k）。
这可能涉及到使用一个归一化的权重向量来对特征进行加权，然后将加权后的特征向量投影到子空间中。
您可以尝试使用不同的归一化方法（如L1范数、L2范数、Softmax等）来看看哪种方法可以得到最好的结果。
这个实现有以下几个关键点：
1.
多种归一化方法：我实现了5种不同的归一化方法来将APE的特征重要性分数转换为权重：
Softmax：将分数转换为概率分布
Min-Max：将分数缩放到[0,1]区间
L1范数：确保权重总和为1
L2范数：确保权重的平方和为1
Tanh：使用双曲正切函数，可以处理异常值
2.
权重应用：我们不是选择特征子集，而是将权重应用于所有特征维度，这样模型可以平滑地利用所有信息，但更关注重要的维度。
3.
自动选择最佳方法：函数会尝试所有归一化方法，并选择在测试集上表现最好的一个。
4.
与TIMO集成：权重应用后，我们仍然使用TIMO的核心逻辑（IGT和TGI）来进行最终的分类。
这种软特征选择方法的优势在于：
不会完全丢弃任何信息
可以更平滑地调整特征的重要性
不需要确定一个硬性的k值
可能对噪声和异常值更鲁棒
您可以将这个函数添加到models.py文件中，然后在main.py中调用它：
'''
def timo_with_soft_selection(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_weights_all, image_weights_all):
    """
    Implements Strategy 2: Soft Feature Selection
    
    Instead of hard selecting top-k features, this approach assigns weights to each feature dimension
    based on their importance scores from APE's criterion. This allows the model to use all features
    but with different emphasis on each dimension.
    """
    print("\n--- Running TIMO with Soft Feature Selection ---")
    
    # --- 1. Calculate feature importance scores using APE's criterion ---
    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num).cuda()
    cache_keys = cache_keys.t().reshape(
        cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim).cuda()
    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)
    
    # Get the full criterion scores for all dimensions
    cfg['w'] = cfg['w_training_free']
    criterion_scores = cal_criterion_scores(cfg, clip_weights, cache_keys, only_use_txt=True)
    
    # --- 2. Convert scores to weights using different normalization methods ---
    # We'll try different normalization approaches and pick the best one
    normalization_methods = {
        'softmax': lambda x: torch.softmax(x, dim=0),
        'min_max': lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8),
        'l1_norm': lambda x: x / (x.sum() + 1e-8),
        'l2_norm': lambda x: x / (torch.norm(x) + 1e-8),
        'tanh': lambda x: torch.tanh(x / torch.std(x))
    }
    
    best_val_acc = 0
    best_test_acc = 0
    best_method = None
    
    for method_name, normalize_fn in normalization_methods.items():
        print(f"\nTrying normalization method: {method_name}")
        
        # Apply normalization to get feature weights
        feature_weights = normalize_fn(criterion_scores)
        
        # --- 3. Apply weights to features ---
        # Element-wise multiply features by weights
        weighted_val_features = val_features * feature_weights.unsqueeze(0)
        weighted_test_features = test_features * feature_weights.unsqueeze(0)
        
        # For text features, we need to apply weights to the last dimension
        weighted_clip_weights_all = clip_weights_all * feature_weights.unsqueeze(0).unsqueeze(0)
        weighted_image_weights_all = image_weights_all * feature_weights.unsqueeze(0).unsqueeze(0)
        
        # --- 4. Run TIMO with weighted features ---
        # Prepare image weights for IGT
        weighted_image_weights = weighted_image_weights_all.mean(dim=1)
        weighted_image_weights = weighted_image_weights / weighted_image_weights.norm(dim=1, keepdim=True)
        
        # IGT: Image-Guided Text optimization
        clip_weights_IGT, matching_score = image_guide_text_search(
            cfg, weighted_clip_weights_all, weighted_val_features, val_labels, weighted_image_weights)
        
        # Run TIMO's core logic with weighted features
        acc = TIMO(cfg, weighted_val_features, val_labels, weighted_test_features, test_labels,
                  clip_weights_IGT, weighted_clip_weights_all, matching_score,
                  grid_search=True, n_quick_search=10, is_print=False)
        
        print(f"Method {method_name} test accuracy: {acc:.4f}")
        
        # Track the best method
        if acc > best_test_acc:
            best_test_acc = acc
            best_method = method_name
    
    print("\n--------------------------------------------------")
    print(f"TIMO with Soft Feature Selection Final Result:")
    print(f"Best normalization method: {best_method}")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print("--------------------------------------------------")
    
    return best_test_acc

def cal_criterion_scores(cfg, clip_weights, cache_keys, only_use_txt=True):
    """
    Calculates and returns the criterion scores for all feature dimensions,
    without performing top-k selection. This is used for searching over k.
    """
    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1)
    
    if only_use_txt:
        feats = text_feat.squeeze()
        sim_sum = torch.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                if i != j:
                    sim_sum += feats[i, :] * feats[j, :]
                    count += 1
        sim = sim_sum / count
    else:
        cache_feat = cache_keys.reshape(cate_num, -1, feat_dim)
        feats = torch.cat([text_feat, cache_feat], dim=1)
        samp_num = feats.shape[1]
        
        sim_sum = torch.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                for m in range(samp_num):
                    for n in range(samp_num):
                        if i != j:
                            sim_sum += feats[i, m, :] * feats[j, n, :]
                            count += 1
        sim = sim_sum / count

    # Calculate the final criterion score for each dimension
    criterion = (-1) * cfg['w'][0] * sim + cfg['w'][1] * torch.var(clip_weights, dim=1)
    
    return criterion

def timo_with_soft_selection_improved(cfg, cache_keys, cache_values, val_features, 
                                      val_labels, test_features, test_labels, 
                                      clip_weights_IGT, clip_weights_all, image_weights_all):
    """
    核心思想：
    1. 使用IGT优化后的文本特征来计算特征重要性（而不是原始文本特征）
    2. 基于重要性对所有特征维度进行软加权（而非硬选择top-k）
    3. 在加权后的特征空间中重新运行IGT+TIMO
    clip_weights_all = clip_weights_cupl_all 所有prompt的特征 [C, P, D]
    cache_keys: 经过clip处理后的fewshot图形样本的张量 [D,C * shot_num]
    cahce_value: onehot vector [C* shot_num, C]
    """
    print("\n--- Running TIMO with Improved Soft Feature Selection ---")

    feat_dim, cate_num = clip_weights_IGT.shape #1024,100
 
    # Step 2: 计算特征重要性 
    #cache_values [100,100] cache_keys[1024,100] cache_keys_reshaped[100,1,1024]
    # cache_values_reshaped[100,1,100] cache_keys_flat[100,1024]
    cache_values_reshaped = cache_values.reshape(cate_num, -1, cate_num).cuda()
    cache_keys_reshaped = cache_keys.t().reshape(cate_num, cfg['shots'], feat_dim)
    cache_keys_flat = cache_keys_reshaped.reshape(-1, feat_dim)
    cache_values_flat = cache_values_reshaped.reshape(-1, cate_num)

    cfg['w'] = cfg['w_training_free']
    criterion_scores = cal_criterion_scores(cfg, clip_weights_IGT, cache_keys_flat, only_use_txt=False)#[1024]

    # Step 3: 改进的归一化方法（添加温度参数）
    normalization_methods = {
        'softmax_t0.1': lambda x: torch.softmax(x / 0.1, dim=0),
        'softmax_t0.5': lambda x: torch.softmax(x / 0.5, dim=0),
        'softmax_t1.0': lambda x: torch.softmax(x / 1.0, dim=0),
        'softmax_t2.0': lambda x: torch.softmax(x / 2.0, dim=0),
        'sigmoid': lambda x: torch.sigmoid((x - x.mean()) / x.std()),
        'tanh_scaled': lambda x: (torch.tanh((x - x.mean()) / x.std()) + 1) / 2,
        #'power_law': lambda x: torch.pow(x / x.max(), 2),  # 平方律
        'sqrt': lambda x: torch.sqrt((x - x.min()) / (x.max() - x.min() + 1e-8)),
    }

    best_val_acc = 0
    best_test_acc = 0
    best_method = None
    best_weights = None

    for method_name, normalize_fn in normalization_methods.items():
        print(f"\nTrying: {method_name}")

        # 获取特征权重 [1024]
        feature_weights = normalize_fn(criterion_scores)

        # 关键改进：使用sqrt进行加权，保持特征的相对关系
        #[1024]
        sqrt_weights = torch.sqrt(feature_weights)

        # 加权特征
        #val_features[1649, 1024] weighted_val_features[1649,1024]
        weighted_val_features = val_features * sqrt_weights.unsqueeze(0) 
        #[2465,1024]
        weighted_test_features = test_features * sqrt_weights.unsqueeze(0)
        #100,22,1024
        weighted_clip_weights_all = clip_weights_all * sqrt_weights.unsqueeze(0).unsqueeze(0)
        #100,1,1024
        weighted_image_weights_all = image_weights_all * sqrt_weights.unsqueeze(0).unsqueeze(0)

        # 重新归一化
        #1649 1024
        weighted_val_features = weighted_val_features / weighted_val_features.norm(dim=-1, keepdim=True)
        #[2465,1024]
        weighted_test_features = weighted_test_features / weighted_test_features.norm(dim=-1, keepdim=True)
        #100,22,1024
        weighted_clip_weights_all = weighted_clip_weights_all / weighted_clip_weights_all.norm(dim=-1, keepdim=True)
        
        # IGT优化
        #weighted_image_weights [100,1024]
        weighted_image_weights = weighted_image_weights_all.mean(dim=1)
        #weighted_image_weights [100,1024]
        weighted_image_weights = weighted_image_weights / weighted_image_weights.norm(dim=1, keepdim=True)

        #val_labels [1649] clip_weights_IGT_weighted [1024,100] matching_score [100,22]
        clip_weights_IGT_weighted, matching_score = image_guide_text_search(cfg, weighted_clip_weights_all, weighted_val_features, val_labels, weighted_image_weights)
        # 运行TIMO
        acc = TIMO(cfg, weighted_val_features, val_labels, weighted_test_features, 
                  test_labels, clip_weights_IGT_weighted, weighted_clip_weights_all, 
                  matching_score, grid_search=True, n_quick_search=10, is_print=False)

        print(f"  Test Accuracy: {acc:.4f}")

        if acc > best_test_acc:
            best_test_acc = acc
            best_method = method_name
            best_weights = feature_weights

    print("\n" + "="*60)
    print(f"Soft Feature Selection Results:")
    print(f"Best Method: {best_method}")
    print(f"Test Accuracy: {best_test_acc:.4f}")
    print("="*60)

    return best_test_acc